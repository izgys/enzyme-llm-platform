"""
src/data/download.py

Downloads reviewed Swiss-Prot entries with EC annotations from the UniProt REST API.

What this script does, in order:
    1. Reads configuration from data_config.yaml
    2. Queries the UniProt API with pagination (500 entries per page)
    3. Extracts the fields we need from each entry's raw JSON
    4. Filters out entries that don't meet our quality criteria
    5. Saves the cleaned records to data/raw/ as JSON
    6. Writes a checksum so we can verify data integrity later

Why we do it this way:
    - Config-driven: no hardcoded parameters
    - Pagination-aware: UniProt won't return 250k entries at once
    - Normalise early: flatten UniProt's nested JSON into our own flat schema
      immediately — downstream code never needs to know UniProt's structure
    - Checksum: anyone can verify they have the same data we trained on
"""

import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# We use logging rather than print() so severity and timestamps are captured
# and log level can be controlled at runtime without changing code
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# EC number extraction
# ---------------------------------------------------------------------------
# This is the non-obvious part of the pipeline.
#
# In UniProt's JSON, EC numbers are NOT a flat field. They live nested inside
# proteinDescription like:
#   entry["proteinDescription"]["recommendedName"]["ecNumbers"][0]["value"]
#
# BUT not all entries have recommendedName, and not all have ecNumbers there.
# We request the "ec" field explicitly via the API fields parameter, which
# UniProt flattens for us. If that's empty, we fall back to proteinDescription.

def extract_ec_numbers(entry: dict) -> list[str]:
    """
    Extract all EC numbers from a UniProt JSON entry.
    Returns a list like ["3.4.21.4", "3.4.21.5"].
    Returns empty list if none found.
    """
    ec_numbers = []

    # Primary path: flat "ec" field we requested directly
    if "ec" in entry and entry["ec"]:
        for ec in str(entry["ec"]).split(","):
            ec = ec.strip()
            if ec:
                ec_numbers.append(ec)

    # Fallback: parse from proteinDescription
    if not ec_numbers and "proteinDescription" in entry:
        desc = entry["proteinDescription"]
        for name_type in ["recommendedName", "submittedName"]:
            if name_type in desc:
                name_entry = desc[name_type]
                if isinstance(name_entry, list):
                    name_entry = name_entry[0]
                for ec_entry in name_entry.get("ecNumbers", []):
                    ec_numbers.append(ec_entry["value"])

    return ec_numbers


def is_valid_ec(ec: str, min_digits: int = 2) -> bool:
    """
    Check EC number has at least min_digits levels of specificity.

    "3.-.-.-" → 1 specific digit → rejected at min_digits=2
    "3.4.-.-" → 2 specific digits → accepted
    "3.4.21.4" → 4 specific digits → accepted

    Why reject incomplete annotations?
    "3.-.-.-" only tells us top-level class (hydrolase) — that's our
    6-class baseline, not the sub-class task we're actually solving.
    """
    parts = ec.split(".")
    specific = sum(1 for p in parts if p.isdigit())
    return specific >= min_digits


def get_ec_subclass(ec: str) -> str:
    """
    Extract 2-digit sub-class label from a full EC number.
    "3.4.21.4" → "3.4"
    """
    parts = ec.split(".")
    return f"{parts[0]}.{parts[1]}"


# ---------------------------------------------------------------------------
# Record normalisation
# ---------------------------------------------------------------------------
# We convert UniProt's nested JSON into our own flat schema immediately.
# Downstream code (dataset.py, clustering.py) never sees UniProt's structure.
#
# Our schema per record:
# {
#     "uniprot_id": str,       e.g. "P00734"
#     "sequence":   str,       amino acid sequence
#     "ec_numbers": list[str], all valid EC numbers
#     "ec_subclass": str,      primary label, e.g. "3.4"
#     "protein_name": str,
#     "organism": str,
#     "length": int,
# }

def normalise_entry(entry: dict, min_ec_digits: int) -> dict | None:
    """
    Convert a raw UniProt entry to our flat schema.
    Returns None if the entry fails quality checks.
    """
    uniprot_id = entry.get("primaryAccession", "")
    if not uniprot_id:
        return None

    seq_data = entry.get("sequence", {})
    sequence = seq_data.get("value", "")
    length = seq_data.get("length", len(sequence))
    if not sequence:
        return None

    ec_numbers = extract_ec_numbers(entry)
    ec_numbers = [ec for ec in ec_numbers if is_valid_ec(ec, min_ec_digits)]
    if not ec_numbers:
        return None

    # Primary label: first EC number's sub-class
    # UniProt orders EC numbers by evidence strength
    ec_subclass = get_ec_subclass(ec_numbers[0])

    # Protein name — navigate nested structure defensively
    protein_name = ""
    try:
        rec_name = entry["proteinDescription"]["recommendedName"]
        if isinstance(rec_name, list):
            rec_name = rec_name[0]
        protein_name = rec_name["fullName"]["value"]
    except (KeyError, TypeError, IndexError):
        protein_name = entry.get("uniProtkbId", "")

    organism = ""
    try:
        organism = entry["organism"]["scientificName"]
    except (KeyError, TypeError):
        pass

    return {
        "uniprot_id": uniprot_id,
        "sequence": sequence,
        "ec_numbers": ec_numbers,
        "ec_subclass": ec_subclass,
        "protein_name": protein_name,
        "organism": organism,
        "length": length,
    }


# ---------------------------------------------------------------------------
# Length filtering
# ---------------------------------------------------------------------------

def passes_length_filter(record: dict, min_length: int, max_length: int) -> bool:
    """
    Filter sequences outside ESM-2's useful range.

    min_length=50:   shorter sequences are likely fragments or signal peptides,
                     not full-length functional enzymes
    max_length=1024: ESM-2's context window limit — the tokeniser silently
                     truncates longer sequences, losing C-terminal information.
                     We exclude rather than silently lose data.
    """
    return min_length <= record["length"] <= max_length


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------
# UniProt returns results in pages of up to 500 entries.
# Each response includes a Link header with rel="next" pointing to the
# next page URL. We follow these links until none remains.
#
# Why this matters: if you only fetch page 1 you get 500 of ~250,000 entries
# and won't know you missed the rest. Silent data loss is the worst kind.

def fetch_all_entries(config: dict) -> list[dict]:
    """
    Paginate through all UniProt results for our query.
    Returns list of raw JSON entry dicts.
    """
    url_config = config["uniprot"]

    params = {
        "query": url_config["query"],
        "format": url_config["format"],
        "fields": ",".join(url_config["fields"]),
        "size": url_config["page_size"],
    }

    all_entries = []
    page = 0
    current_url = url_config["base_url"]
    current_params = params

    pbar = tqdm(desc="Downloading UniProt entries", unit=" entries")

    while True:
        page += 1

        # Retry logic: UniProt occasionally returns 429 (rate limited) or 503
        for attempt in range(3):
            try:
                response = requests.get(
                    current_url,
                    params=current_params if page == 1 else None,
                    timeout=60,
                )
                response.raise_for_status()
                break
            except requests.exceptions.HTTPError:
                if response.status_code == 429:
                    wait = int(response.headers.get("Retry-After", 5))
                    logger.warning(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                elif attempt == 2:
                    raise
                else:
                    time.sleep(2 ** attempt)

        data = response.json()
        entries = data.get("results", [])
        all_entries.extend(entries)
        pbar.update(len(entries))

        # Check for next page
        # Link header format: <https://...>; rel="next"
        link_header = response.headers.get("Link", "")
        if 'rel="next"' not in link_header:
            break  # Last page

        # Extract next URL — params are embedded in it, don't pass them again
        current_url = link_header.split("<")[1].split(">")[0]
        current_params = None

        time.sleep(0.1)  # Be a polite API consumer

    pbar.close()
    logger.info(f"Downloaded {len(all_entries):,} raw entries across {page} pages")
    return all_entries


# ---------------------------------------------------------------------------
# Checksum
# ---------------------------------------------------------------------------
# MD5 of the output file lets anyone verify they have the same data we used.
# This is what makes the data versioning claim in the README real.

def compute_checksum(filepath: Path) -> str:
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config_path: str = "config/data_config.yaml") -> None:
    config = load_config(config_path)
    filtering = config["filtering"]
    output_cfg = config["output"]

    raw_dir = Path(output_cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / output_cfg["raw_filename"]
    checksums_path = Path(output_cfg["checksums_file"])

    # Download
    logger.info("Starting UniProt download...")
    logger.info(f"Query: {config['uniprot']['query']}")
    raw_entries = fetch_all_entries(config)

    # Normalise and filter
    logger.info("Normalising and filtering entries...")
    records = []
    stats = {
        "total_raw": len(raw_entries),
        "no_ec": 0,
        "length_filtered": 0,
        "accepted": 0,
    }

    for entry in tqdm(raw_entries, desc="Processing entries"):
        record = normalise_entry(entry, filtering["min_ec_digits"])
        if record is None:
            stats["no_ec"] += 1
            continue
        if not passes_length_filter(
            record, filtering["min_length"], filtering["max_length"]
        ):
            stats["length_filtered"] += 1
            continue
        records.append(record)
        stats["accepted"] += 1

    # Report
    logger.info("=" * 50)
    logger.info(f"Raw entries downloaded:   {stats['total_raw']:>8,}")
    logger.info(f"Dropped (no valid EC):    {stats['no_ec']:>8,}")
    logger.info(f"Dropped (length filter):  {stats['length_filtered']:>8,}")
    logger.info(f"Accepted records:         {stats['accepted']:>8,}")
    logger.info("=" * 50)

    subclass_counts = Counter(r["ec_subclass"] for r in records)
    logger.info(f"Unique EC sub-classes:    {len(subclass_counts):>8,}")
    logger.info("Top 10 sub-classes by count:")
    for subclass, count in subclass_counts.most_common(10):
        logger.info(f"  EC {subclass:<10} {count:>6,} sequences")

    # Save
    logger.info(f"Saving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    # Checksum
    checksum = compute_checksum(output_path)
    checksums = {}
    if checksums_path.exists():
        with open(checksums_path) as f:
            checksums = json.load(f)
    checksums[output_cfg["raw_filename"]] = {
        "md5": checksum,
        "n_records": len(records),
        "n_subclasses": len(subclass_counts),
    }
    checksums_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checksums_path, "w") as f:
        json.dump(checksums, f, indent=2)

    logger.info(f"Checksum written to {checksums_path}")
    logger.info("Download complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Download Swiss-Prot enzyme entries from UniProt"
    )
    parser.add_argument(
        "--config",
        default="config/data_config.yaml",
        help="Path to data config YAML",
    )
    args = parser.parse_args()
    run(args.config)