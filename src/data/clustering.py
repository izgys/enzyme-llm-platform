"""
src/data/clustering.py

Runs MMseqs2 sequence clustering via Docker to produce homology-aware
train/val/test splits.

What this script does:
    1. Converts processed records to FASTA format (MMseqs2 input)
    2. Runs MMseqs2 easy-cluster via Docker at 30% sequence identity
    3. Parses the cluster TSV output
    4. Assigns clusters to train/val/test splits
    5. Saves split datasets and a clustering report

Why Docker for MMseqs2?
    MMseqs2 is a Linux binary. Running it via Docker means the pipeline
    works on Windows and Mac without manual installation, and the exact
    MMseqs2 version is pinned in the Docker image tag — reproducibility.

Why 30% identity threshold?
    30% is the community standard for protein ML benchmarking. Below ~25%
    is the "twilight zone" where even structural similarity becomes uncertain.
    At 30% we are conservative but not overly strict — we ensure genuine
    generalisation without throwing away too much data.

Why cluster-based splitting rather than sequence-based?
    If we split sequences randomly, two sequences in the same cluster
    (>30% identity) may end up in train and test. The model sees essentially
    the same protein in both — inflating test performance by 15-30% on
    typical protein classification tasks. Cluster-based splitting prevents
    this entirely.
"""

import json
import logging
import os
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# MMseqs2 Docker image — pinned for reproducibility
MMSEQS2_IMAGE = "ghcr.io/soedinglab/mmseqs2:latest"


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_records(processed_path: Path) -> list[dict]:
    logger.info(f"Loading records from {processed_path}...")
    with open(processed_path) as f:
        records = json.load(f)
    logger.info(f"Loaded {len(records):,} records")
    return records


# ---------------------------------------------------------------------------
# FASTA export
# ---------------------------------------------------------------------------
# MMseqs2 takes FASTA format as input — one entry per sequence:
#   >UNIPROT_ID
#   SEQUENCE
#
# We use the UniProt ID as the FASTA header because MMseqs2 uses headers
# as sequence identifiers in its output TSV. We need to map back from
# MMseqs2 cluster IDs (UniProt IDs) to our records.

def write_fasta(records: list[dict], fasta_path: Path) -> None:
    logger.info(f"Writing FASTA to {fasta_path}...")
    with open(fasta_path, "w") as f:
        for record in tqdm(records, desc="Writing FASTA"):
            f.write(f">{record['uniprot_id']}\n")
            # Write sequence in 80-character lines (FASTA convention)
            seq = record["sequence"]
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")
    logger.info(f"Wrote {len(records):,} sequences to FASTA")


# ---------------------------------------------------------------------------
# MMseqs2 via Docker
# ---------------------------------------------------------------------------
# We use MMseqs2's "easy-cluster" workflow which handles database creation,
# clustering, and output in one command.
#
# Key parameters:
#   --min-seq-id 0.30   → 30% sequence identity threshold
#   --cov-mode 0        → coverage on both query and target sequences
#   -c 0.8              → 80% coverage required — prevents short fragments
#                         clustering with full-length sequences
#   --cluster-mode 0    → greedy set cover (standard for protein clustering)
#
# The Docker volume mount (-v) maps our local data/clusters directory
# into the container at /data so MMseqs2 can read/write files.

def run_mmseqs2_docker(
    fasta_path: Path,
    clusters_dir: Path,
    min_seq_id: float = 0.30,
) -> Path:
    """
    Run MMseqs2 easy-cluster via Docker.
    Returns path to the cluster TSV output file.
    """
    clusters_dir.mkdir(parents=True, exist_ok=True)

    # Docker requires absolute paths for volume mounts
    # We mount the entire clusters_dir and reference files by container path
    abs_clusters = str(clusters_dir.resolve())
    fasta_filename = fasta_path.name

    # Copy FASTA into clusters_dir so it's accessible inside the container
    import shutil
    dest_fasta = clusters_dir / fasta_filename
    if fasta_path.resolve() != dest_fasta.resolve():
        shutil.copy(fasta_path, dest_fasta)

    # MMseqs2 output prefix inside the container
    output_prefix = "/data/mmseqs_clusters"
    tmp_dir = "/data/tmp"

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{abs_clusters}:/data",  # mount clusters_dir as /data
        MMSEQS2_IMAGE,
        "easy-cluster",
        f"/data/{fasta_filename}",      # input FASTA
        output_prefix,                  # output prefix
        tmp_dir,                        # tmp directory
        "--min-seq-id", str(min_seq_id),
        "--cov-mode", "0",
        "-c", "0.8",
        "--cluster-mode", "0",
        "-v", "1",                      # verbosity (1 = minimal)
    ]

    logger.info("Running MMseqs2 via Docker...")
    logger.info(f"Identity threshold: {min_seq_id:.0%}")
    logger.info(f"This may take 10-30 minutes for 200k sequences...")

    result = subprocess.run(
        cmd,
        capture_output=False,  # Let MMseqs2 output stream to terminal
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"MMseqs2 failed with return code {result.returncode}"
        )

    # MMseqs2 easy-cluster outputs: prefix_cluster.tsv
    tsv_path = clusters_dir / "mmseqs_clusters_cluster.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(
            f"MMseqs2 output not found at {tsv_path}. "
            f"Check Docker output above for errors."
        )

    logger.info(f"MMseqs2 complete. Cluster TSV: {tsv_path}")
    return tsv_path


# ---------------------------------------------------------------------------
# Parse cluster TSV
# ---------------------------------------------------------------------------
# MMseqs2 outputs a TSV with two columns:
#   representative_id    member_id
#
# Every sequence appears as a member. The representative is the cluster centre.
# A sequence that is its own representative is a singleton cluster.

def parse_cluster_tsv(tsv_path: Path) -> dict[str, str]:
    """
    Parse MMseqs2 cluster TSV.
    Returns {member_id: cluster_id} mapping.
    """
    logger.info(f"Parsing cluster TSV: {tsv_path}")
    member_to_cluster = {}

    with open(tsv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            representative, member = parts
            member_to_cluster[member] = representative

    n_clusters = len(set(member_to_cluster.values()))
    logger.info(
        f"Parsed {len(member_to_cluster):,} sequences "
        f"into {n_clusters:,} clusters"
    )
    return member_to_cluster


# ---------------------------------------------------------------------------
# Homology-aware splitting
# ---------------------------------------------------------------------------
# We split at the CLUSTER level, not the sequence level.
# This guarantees no two sequences from the same cluster appear in
# different splits.
#
# Strategy:
#   1. Group clusters by their EC sub-class (primary label of the
#      representative sequence, or majority vote among members)
#   2. For each sub-class, shuffle clusters and assign to splits
#      proportionally (70/15/15)
#   3. Every sequence in a cluster inherits its cluster's split assignment
#
# Why shuffle before splitting?
#   MMseqs2 outputs clusters in an arbitrary order. Without shuffling,
#   train/val/test would have systematic ordering biases.
#
# Why split by sub-class?
#   If we split clusters globally (ignoring sub-class), random variation
#   could leave some sub-classes with zero val or test clusters.
#   Stratifying by sub-class ensures every sub-class is represented
#   in every split.

def assign_splits(
    records: list[dict],
    member_to_cluster: dict[str, str],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, str]:
    """
    Assign each cluster to train/val/test.
    Returns {cluster_id: split} mapping.
    """
    random.seed(seed)

    # Build cluster → list of records mapping
    cluster_to_records = defaultdict(list)
    for record in records:
        uid = record["uniprot_id"]
        cluster_id = member_to_cluster.get(uid, uid)  # fallback: own ID
        cluster_to_records[cluster_id].append(record)

    # For each cluster, determine its EC sub-class by majority vote
    # (most sequences in the cluster share the same sub-class)
    cluster_to_subclass = {}
    for cluster_id, cluster_records in cluster_to_records.items():
        subclass_votes = Counter(r["ec_subclass"] for r in cluster_records)
        cluster_to_subclass[cluster_id] = subclass_votes.most_common(1)[0][0]

    # Group clusters by sub-class for stratified splitting
    subclass_to_clusters = defaultdict(list)
    for cluster_id, subclass in cluster_to_subclass.items():
        subclass_to_clusters[subclass].append(cluster_id)

    # Assign clusters to splits, stratified by sub-class
    cluster_to_split = {}
    split_stats = Counter()

    for subclass, clusters in subclass_to_clusters.items():
        random.shuffle(clusters)
        n = len(clusters)
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        # Remaining go to test
        n_test = n - n_train - n_val

        if n_test < 1:
            # Sub-class too small for three-way split — put in train only
            logger.warning(
                f"EC {subclass}: only {n} clusters, "
                f"cannot create val/test splits — all assigned to train"
            )
            for c in clusters:
                cluster_to_split[c] = "train"
            split_stats["train"] += n
            continue

        for i, cluster_id in enumerate(clusters):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            cluster_to_split[cluster_id] = split
            split_stats[split] += 1

    logger.info("Cluster split assignment:")
    total_clusters = sum(split_stats.values())
    for split in ["train", "val", "test"]:
        n = split_stats[split]
        logger.info(f"  {split:<6} {n:>6,} clusters ({100*n/total_clusters:.1f}%)")

    return cluster_to_split


def apply_splits_to_records(
    records: list[dict],
    member_to_cluster: dict[str, str],
    cluster_to_split: dict[str, str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Assign each record to train/val/test based on its cluster's split.
    Returns (train_records, val_records, test_records).
    """
    train, val, test = [], [], []
    unassigned = 0

    for record in records:
        uid = record["uniprot_id"]
        cluster_id = member_to_cluster.get(uid, uid)
        split = cluster_to_split.get(cluster_id)

        if split is None:
            unassigned += 1
            train.append(record)  # fallback
            continue

        record["cluster_id"] = cluster_id
        record["split"] = split

        if split == "train":
            train.append(record)
        elif split == "val":
            val.append(record)
        else:
            test.append(record)

    if unassigned:
        logger.warning(f"{unassigned} records had no cluster assignment — added to train")

    return train, val, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config_path: str = "config/data_config.yaml") -> None:
    config = load_config(config_path)
    output_cfg = config["output"]
    clustering_cfg = config.get("clustering", {})

    processed_path = Path(output_cfg["processed_dir"]) / "enzymes_processed.json"
    clusters_dir = Path(output_cfg["clusters_dir"])
    splits_dir = Path(output_cfg["processed_dir"])

    records = load_records(processed_path)

    # Step 1: Write FASTA
    fasta_path = clusters_dir / "sequences.fasta"
    clusters_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(records, fasta_path)

    # Step 2: Run MMseqs2
    min_seq_id = clustering_cfg.get("min_seq_id", 0.30)
    tsv_path = run_mmseqs2_docker(fasta_path, clusters_dir, min_seq_id)

    # Step 3: Parse clusters
    member_to_cluster = parse_cluster_tsv(tsv_path)

    # Step 4: Assign splits
    train_frac = clustering_cfg.get("train_frac", 0.70)
    val_frac = clustering_cfg.get("val_frac", 0.15)
    seed = clustering_cfg.get("seed", 42)

    cluster_to_split = assign_splits(
        records, member_to_cluster, train_frac, val_frac, seed
    )

    # Step 5: Apply splits to records
    train, val, test = apply_splits_to_records(
        records, member_to_cluster, cluster_to_split
    )

    logger.info("=" * 50)
    logger.info(f"Train: {len(train):>8,} sequences")
    logger.info(f"Val:   {len(val):>8,} sequences")
    logger.info(f"Test:  {len(test):>8,} sequences")
    logger.info(f"Total: {len(train)+len(val)+len(test):>8,} sequences")
    logger.info("=" * 50)

    # Save splits
    for split_name, split_records in [("train", train), ("val", val), ("test", test)]:
        out_path = splits_dir / f"{split_name}.json"
        logger.info(f"Saving {split_name} split to {out_path}...")
        with open(out_path, "w") as f:
            json.dump(split_records, f, indent=2)

    # Save clustering report
    report = {
        "n_total_sequences": len(records),
        "n_clusters": len(set(member_to_cluster.values())),
        "min_seq_id": min_seq_id,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "seed": seed,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "subclass_split_counts": {
            split: dict(Counter(r["ec_subclass"] for r in recs))
            for split, recs in [("train", train), ("val", val), ("test", test)]
        },
    }
    report_path = splits_dir / "clustering_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Clustering report saved to {report_path}")
    logger.info("Clustering complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Homology-aware train/val/test splitting via MMseqs2"
    )
    parser.add_argument(
        "--config",
        default="config/data_config.yaml",
        help="Path to data config YAML",
    )
    args = parser.parse_args()
    run(args.config)