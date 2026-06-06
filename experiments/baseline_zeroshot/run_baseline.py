"""
Zero-shot ESM-2 baseline for enzyme sub-class classification.

What this script does:
    1. Loads ESM-2 650M with frozen weights — no fine-tuning
    2. Extracts CLS token embeddings for train and test sequences
    3. Fits a logistic regression on train embeddings
    4. Evaluates on test embeddings with macro-F1 and per-class F1
    5. Saves results to experiments/baseline_zeroshot/

Why this comes before fine-tuning:
    Without a baseline, fine-tuning results have no reference point.
    The baseline answers: how much does fine-tuning actually help?

Why logistic regression (linear probe)?
    A linear model measures what's already in the representations —
    not what a neural network can learn on top of them. If ESM-2
    embeddings are good, a linear probe will find the structure.
    This is the standard evaluation protocol in the PLM literature.

Runtime estimate:
    Embedding extraction: ~20-40 minutes for 180k sequences on RTX 4070
    Logistic regression: ~5-10 minutes
    Total: ~30-50 minutes
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
)
from torch.utils.data import DataLoader
from transformers import EsmModel, EsmTokenizer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("experiments/baseline_zeroshot")
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
BATCH_SIZE = 16       # sequences per batch — adjust down if OOM
MAX_LENGTH = 512      # use 512 for speed; 1024 for full coverage
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(
    split_path: Path,
    model: EsmModel,
    tokenizer: EsmTokenizer,
    batch_size: int,
    max_length: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract CLS token embeddings for all sequences in a split.

    Returns:
        embeddings: shape (n_sequences, hidden_size) — float32 numpy array
        labels:     shape (n_sequences,) — integer numpy array
    """
    with open(split_path) as f:
        records = json.load(f)

    sequences = [r["sequence"] for r in records]
    labels = np.array([r["label"] for r in records])

    all_embeddings = []

    # Process in batches — we can't fit 150k sequences in GPU memory at once
    for i in tqdm(
        range(0, len(sequences), batch_size),
        desc=f"Extracting embeddings ({split_path.stem})",
    ):
        batch_seqs = sequences[i: i + batch_size]

        # Tokenise batch
        encoding = tokenizer(
            batch_seqs,
            max_length=max_length,
            padding=True,        # pad to longest in batch (not max_length)
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Forward pass — no gradients needed, saves memory and compute
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Extract CLS token embedding — position 0 in the sequence
        # Shape: (batch_size, hidden_size) = (batch_size, 1280)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        # Move to CPU and convert to numpy — GPU memory is precious
        all_embeddings.append(cls_embeddings.cpu().float().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(
        f"Extracted embeddings: {embeddings.shape} "
        f"for {len(labels)} sequences"
    )
    return embeddings, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Max length: {MAX_LENGTH}")

    # Load label map for readable class names in the report
    with open(PROCESSED_DIR / "label_map.json") as f:
        label_map = json.load(f)
    # Invert: {integer: ec_subclass_string}
    idx_to_subclass = {v: k for k, v in label_map.items()}

    # Load ESM-2 — frozen, no classification head
    # We use EsmModel (not EsmForSequenceClassification) because we want
    # raw embeddings, not logits
    logger.info(f"Loading {MODEL_NAME}...")
    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)
    model.eval()        # disable dropout
    model.to(DEVICE)

    # Count parameters — good to know and good to mention in interviews
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"ESM-2 parameters: {n_params:,} (all frozen)")

    # Extract embeddings
    t0 = time.time()
    train_embeddings, train_labels = extract_embeddings(
        PROCESSED_DIR / "train.json",
        model, tokenizer, BATCH_SIZE, MAX_LENGTH, DEVICE,
    )
    val_embeddings, val_labels = extract_embeddings(
        PROCESSED_DIR / "val.json",
        model, tokenizer, BATCH_SIZE, MAX_LENGTH, DEVICE,
    )
    test_embeddings, test_labels = extract_embeddings(
        PROCESSED_DIR / "test.json",
        model, tokenizer, BATCH_SIZE, MAX_LENGTH, DEVICE,
    )
    embedding_time = time.time() - t0
    logger.info(f"Embedding extraction: {embedding_time/60:.1f} minutes")

    # Save embeddings — expensive to recompute, useful for future experiments
    logger.info("Saving embeddings...")
    np.save(OUTPUT_DIR / "train_embeddings.npy", train_embeddings)
    np.save(OUTPUT_DIR / "train_labels.npy", train_labels)
    np.save(OUTPUT_DIR / "val_embeddings.npy", val_embeddings)
    np.save(OUTPUT_DIR / "val_labels.npy", val_labels)
    np.save(OUTPUT_DIR / "test_embeddings.npy", test_embeddings)
    np.save(OUTPUT_DIR / "test_labels.npy", test_labels)

    # Free GPU memory before sklearn takes over
    del model
    torch.cuda.empty_cache()

    # Fit logistic regression on train embeddings
    logger.info("Fitting logistic regression...")
    logger.info("(This may take 5-10 minutes for 150k x 1280 embeddings)")
    t0 = time.time()

    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,              # regularisation strength — standard default
        solver="lbfgs",     # works well for multiclass, memory efficient
        n_jobs=-1,          # use all CPU cores
        verbose=1,
    )
    clf.fit(train_embeddings, train_labels)
    lr_time = time.time() - t0
    logger.info(f"Logistic regression fit: {lr_time/60:.1f} minutes")

    # Evaluate
    logger.info("Evaluating on test set...")
    test_preds = clf.predict(test_embeddings)

    macro_f1 = f1_score(test_labels, test_preds, average="macro")
    micro_f1 = f1_score(test_labels, test_preds, average="micro")

    logger.info("=" * 50)
    logger.info(f"Zero-shot ESM-2 baseline results:")
    logger.info(f"  Macro-F1:  {macro_f1:.4f}")
    logger.info(f"  Micro-F1:  {micro_f1:.4f}")
    logger.info("=" * 50)

    # Per-class report
    target_names = [idx_to_subclass[i] for i in sorted(idx_to_subclass.keys())]
    report = classification_report(
        test_labels,
        test_preds,
        target_names=target_names,
        digits=3,
    )
    logger.info(f"\nPer-class report:\n{report}")

    # Also evaluate on val set — useful for comparing with fine-tuned model
    val_preds = clf.predict(val_embeddings)
    val_macro_f1 = f1_score(val_labels, val_preds, average="macro")
    logger.info(f"Val Macro-F1: {val_macro_f1:.4f}")

    # Save results
    results = {
        "model": MODEL_NAME,
        "probe": "LogisticRegression",
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "test_macro_f1": float(macro_f1),
        "test_micro_f1": float(micro_f1),
        "val_macro_f1": float(val_macro_f1),
        "embedding_time_minutes": round(embedding_time / 60, 1),
        "lr_fit_time_minutes": round(lr_time / 60, 1),
        "n_train": len(train_labels),
        "n_val": len(val_labels),
        "n_test": len(test_labels),
        "n_classes": len(label_map),
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(OUTPUT_DIR / "per_class_report.txt", "w") as f:
        f.write(report)

    logger.info(f"Results saved to {OUTPUT_DIR}")
    logger.info("Zero-shot baseline complete.")


if __name__ == "__main__":
    run()