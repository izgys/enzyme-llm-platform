"""
src/data/dataset.py

PyTorch Dataset class for enzyme family classification with ESM-2.

What this module does:
    - Loads train/val/test split JSON files
    - Tokenises amino acid sequences using ESM-2's tokeniser
    - Returns (input_ids, attention_mask, label) tensors for each record

Key design decisions:
    - Tokenisation happens at __getitem__ time, not at load time
      Reason: loading all 150k sequences pre-tokenised would require
      ~10GB RAM. On-the-fly tokenisation costs negligible compute and
      keeps memory footprint small.
    - Max length capped at 1024 (ESM-2 context window)
      Sequences longer than this were already filtered in preprocessing,
      but we truncate defensively here too.
    - We return attention_mask explicitly
      The DataLoader collate function needs it to pad batches correctly.
"""

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import EsmTokenizer

logger = logging.getLogger(__name__)


class EnzymeDataset(Dataset):
    """
    PyTorch Dataset for enzyme EC sub-class classification.

    Args:
        split_path:   Path to train.json, val.json, or test.json
        tokenizer:    ESM-2 tokeniser (EsmTokenizer)
        max_length:   Maximum sequence length — ESM-2 context window is 1024
    """

    def __init__(
        self,
        split_path: str | Path,
        tokenizer: EsmTokenizer,
        max_length: int = 1024,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

        split_path = Path(split_path)
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        logger.info(f"Loading dataset from {split_path}...")
        with open(split_path) as f:
            self.records = json.load(f)
        logger.info(f"Loaded {len(self.records):,} records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        sequence = record["sequence"]
        label = record["label"]

        # Tokenise the sequence
        # What the tokeniser does:
        #   1. Maps each amino acid character to an integer ID
        #   2. Prepends [CLS] token (id=0 in ESM-2 vocabulary)
        #   3. Appends [EOS] token
        #   4. Pads or truncates to max_length
        #   5. Builds attention_mask: 1 for real tokens, 0 for padding
        encoding = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding="max_length",   # pad all sequences to max_length
            truncation=True,        # truncate if longer than max_length
            return_tensors="pt",    # return PyTorch tensors
        )

        return {
            # input_ids: integer token IDs, shape (max_length,)
            "input_ids": encoding["input_ids"].squeeze(0),
            # attention_mask: 1 for real tokens, 0 for padding, shape (max_length,)
            "attention_mask": encoding["attention_mask"].squeeze(0),
            # label: integer class index
            "label": torch.tensor(label, dtype=torch.long),
        }

    def get_class_counts(self) -> dict[int, int]:
        """
        Count sequences per class label.
        Used for computing class weights for weighted sampling or loss.
        """
        from collections import Counter
        return dict(Counter(r["label"] for r in self.records))

    def get_label_to_subclass(self) -> dict[int, str]:
        """
        Build {integer_label: ec_subclass_string} mapping from records.
        Useful for converting model predictions back to EC sub-class strings.
        """
        return {
            r["label"]: r["ec_subclass"]
            for r in self.records
        }


def create_dataloaders(
    data_dir: str | Path,
    tokenizer: EsmTokenizer,
    batch_size: int = 16,
    max_length: int = 1024,
    num_workers: int = 0,
) -> tuple:
    """
    Create train, val, and test DataLoaders.

    Args:
        data_dir:    Directory containing train.json, val.json, test.json
        tokenizer:   ESM-2 tokeniser
        batch_size:  Sequences per batch — keep low (8-16) for 8GB VRAM
        max_length:  ESM-2 context window
        num_workers: Parallel data loading workers
                     Note: set to 0 on Windows — multiprocessing with
                     PyTorch DataLoader has known issues on Windows

    Returns:
        (train_loader, val_loader, test_loader, train_dataset)
        We return train_dataset separately for class weight computation.
    """
    data_dir = Path(data_dir)

    train_dataset = EnzymeDataset(data_dir / "train.json", tokenizer, max_length)
    val_dataset = EnzymeDataset(data_dir / "val.json", tokenizer, max_length)
    test_dataset = EnzymeDataset(data_dir / "test.json", tokenizer, max_length)

    # Why shuffle=True for train only?
    # Training: shuffling prevents the model from learning order-based
    # patterns and ensures each batch has a mix of classes.
    # Val/test: we never shuffle — results must be deterministic and
    # reproducible for fair comparison across runs.
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,   # faster GPU transfer
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_dataset