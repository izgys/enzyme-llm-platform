"""
src/data/preprocess.py

Preprocesses the raw UniProt download for model training.

What this script does:
    1. Loads the raw JSON from download.py
    2. Deduplicates at 100% sequence identity
    3. Analyses class distribution and drops sub-classes below min_count
    4. Builds and saves a label map (ec_subclass → integer index)
    5. Saves the processed dataset ready for clustering

Why deduplication before clustering?
    MMseqs2 clustering is the expensive step. Removing exact duplicates first
    reduces that cost and avoids identical sequences appearing in both train
    and test sets even after clustering (100% identity is always in the same
    cluster, but why pay for it).

Why drop rare sub-classes?
    A sub-class with 10 sequences cannot be split into train/val/test and
    evaluated meaningfully. We set a minimum count threshold and either drop
    or merge rare classes. Dropped classes are documented — we never silently
    discard data.
"""

import json
import logging
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_records(raw_path: Path) -> list[dict]:
    logger.info(f"Loading records from {raw_path}...")
    with open(raw_path) as f:
        records = json.load(f)
    logger.info(f"Loaded {len(records):,} records")
    return records


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
# We deduplicate on exact sequence identity (100%).
# Why keep the first occurrence? UniProt orders entries by annotation quality
# within a query — reviewed entries with more evidence come first.

def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    """
    Remove records with identical sequences, keeping first occurrence.
    Returns (deduplicated_records, n_dropped).
    """
    seen_sequences = set()
    unique_records = []

    for record in tqdm(records, desc="Deduplicating"):
        seq = record["sequence"]
        if seq not in seen_sequences:
            seen_sequences.add(seq)
            unique_records.append(record)

    n_dropped = len(records) - len(unique_records)
    return unique_records, n_dropped


# ---------------------------------------------------------------------------
# Class distribution analysis
# ---------------------------------------------------------------------------

def analyse_distribution(records: list[dict]) -> Counter:
    return Counter(r["ec_subclass"] for r in records)


def print_distribution(counts: Counter, title: str = "EC sub-class distribution") -> None:
    logger.info(f"\n{title}")
    logger.info(f"{'Sub-class':<12} {'Count':>8} {'%':>6}")
    logger.info("-" * 30)
    total = sum(counts.values())
    for subclass, count in sorted(counts.items()):
        pct = 100 * count / total
        logger.info(f"  EC {subclass:<8} {count:>8,} {pct:>5.1f}%")
    logger.info(f"  {'TOTAL':<8} {total:>8,} 100.0%")
    logger.info(f"  Unique sub-classes: {len(counts)}")


# ---------------------------------------------------------------------------
# Rare class handling
# ---------------------------------------------------------------------------
# We drop sub-classes with fewer than min_count sequences.
#
# Why 100 as default?
#   With homology-aware splitting, effective sequences per class can be
#   much lower than raw counts (recall the aminoacyl-tRNA synthetase insight).
#   A sub-class with 100 raw sequences might have only 20-30 independent
#   clusters — barely enough for train/val/test. Below 100 raw sequences,
#   per-class F1 estimates become unreliable.
#
# Dropped classes are always logged — never silently discarded.

def filter_rare_classes(
    records: list[dict], min_count: int
) -> tuple[list[dict], list[str]]:
    """
    Drop records from sub-classes with fewer than min_count sequences.
    Returns (filtered_records, list_of_dropped_subclasses).
    """
    counts = analyse_distribution(records)
    rare = {sc for sc, n in counts.items() if n < min_count}

    if rare:
        logger.warning(
            f"Dropping {len(rare)} sub-classes with fewer than {min_count} sequences:"
        )
        for sc in sorted(rare):
            logger.warning(f"  EC {sc}: {counts[sc]} sequences")

    filtered = [r for r in records if r["ec_subclass"] not in rare]
    return filtered, sorted(rare)


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------
# The model needs integer labels, not strings like "3.4".
# We build a deterministic map: sort sub-classes alphabetically so the
# mapping is stable across runs, then save it to disk.
#
# Why save it?
#   At inference time, you need to decode integer predictions back to
#   EC sub-class strings. If you rebuild the map on a different dataset
#   or in a different order, the integers mean different things.
#   The saved label map is part of the model artifact.

def build_label_map(records: list[dict]) -> dict[str, int]:
    """
    Build {ec_subclass: integer_index} map, sorted alphabetically for stability.
    """
    subclasses = sorted(set(r["ec_subclass"] for r in records))
    return {sc: idx for idx, sc in enumerate(subclasses)}


def apply_labels(records: list[dict], label_map: dict[str, int]) -> list[dict]:
    """Add integer label field to each record."""
    for record in records:
        record["label"] = label_map[record["ec_subclass"]]
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config_path: str = "config/data_config.yaml") -> None:
    config = load_config(config_path)
    filtering = config["filtering"]
    output_cfg = config["output"]

    raw_path = Path(output_cfg["raw_dir"]) / output_cfg["raw_filename"]
    processed_dir = Path(output_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load
    records = load_records(raw_path)

    # Deduplicate
    records, n_dupes = deduplicate(records)
    logger.info(f"Removed {n_dupes:,} exact duplicates → {len(records):,} unique sequences")

    # Distribution before filtering
    counts_before = analyse_distribution(records)
    print_distribution(counts_before, "Distribution before rare-class filtering")

    # Drop rare classes
    min_count = filtering.get("min_class_count", 100)
    records, dropped_classes = filter_rare_classes(records, min_count)
    logger.info(f"After rare-class filtering: {len(records):,} records")

    # Distribution after filtering
    counts_after = analyse_distribution(records)
    print_distribution(counts_after, "Distribution after rare-class filtering")

    # Build label map
    label_map = build_label_map(records)
    logger.info(f"Label map: {len(label_map)} classes")
    logger.info(f"Classes: {list(label_map.keys())}")

    # Apply integer labels
    records = apply_labels(records, label_map)

    # Save processed records
    processed_path = processed_dir / "enzymes_processed.json"
    logger.info(f"Saving processed records to {processed_path}...")
    with open(processed_path, "w") as f:
        json.dump(records, f, indent=2)

    # Save label map — this is part of the model artifact
    label_map_path = processed_dir / "label_map.json"
    logger.info(f"Saving label map to {label_map_path}...")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    # Save preprocessing report
    report = {
        "n_raw": len(records) + n_dupes,
        "n_after_dedup": len(records) + len(dropped_classes),
        "n_duplicates_removed": n_dupes,
        "n_rare_classes_dropped": len(dropped_classes),
        "dropped_classes": dropped_classes,
        "n_final_records": len(records),
        "n_classes": len(label_map),
        "min_class_count": min_count,
        "class_counts": dict(counts_after.most_common()),
    }
    report_path = processed_dir / "preprocessing_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {report_path}")
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess UniProt enzyme records")
    parser.add_argument(
        "--config",
        default="config/data_config.yaml",
        help="Path to data config YAML",
    )
    args = parser.parse_args()
    run(args.config)