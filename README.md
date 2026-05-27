# ESM2-EnzymeClassifier — Protein Language Model Fine-Tuning for Enzyme Family Classification

**Author:** Iker Zapirain Gysling  
**Status:** In development  
**Domain:** Protein Engineering · Computational Biology · Production ML

---

## The Problem

Enzyme family classification from sequence is a foundational task in computational biology and drug discovery. Knowing which enzyme family a protein belongs to informs target identification, selectivity profiling, off-target risk assessment, and the design of selective inhibitors. Classical approaches rely on sequence similarity (BLAST, HMMs) and struggle with remote homologues — proteins that share a functional fold but have diverged beyond recognisable sequence identity.

Protein language models (PLMs) such as ESM-2 learn deep representations of protein sequences from evolutionary co-variation across hundreds of millions of sequences. These representations capture functional and structural information without explicit structural input, offering a principled route to classification that generalises beyond shallow sequence similarity.

This project addresses four concrete scientific questions:

- How well do ESM-2 embeddings encode enzyme family identity, and how much does fine-tuning improve over zero-shot classification?
- How should training and validation sets be split to prevent homology leakage — the main source of inflated performance in protein ML benchmarks?
- How calibrated are the model's confidence scores, and when should predictions be trusted?
- Which sequence positions drive classification decisions, and do they correspond to functionally important residues?

---

## What It Does

This repository fine-tunes ESM-2 (650M parameter) with Low-Rank Adaptation (LoRA) on enzyme family classification from UniProtKB/Swiss-Prot. The full system includes:

- Homology-aware train/validation/test splitting (sequence identity clustering via MMseqs2)
- LoRA-based parameter-efficient fine-tuning (< 1% of ESM-2 parameters updated)
- Calibrated confidence scores with explicit uncertainty quantification
- Integrated gradients interpretability mapped to sequence positions, cross-validated against MSA conservation
- Structure mapping of attribution scores to PDB entries
- FastAPI inference endpoint with Docker packaging
- MLflow experiment tracking across all runs

---

## Architecture

```
UniProtKB/Swiss-Prot
        │
        │  ETL pipeline (filtering, deduplication, label encoding)
        ▼
Sequence Dataset
        │
        │  MMseqs2 clustering (homology-aware splitting)
        ▼
Train / Val / Test splits
        │
        ▼
ESM-2 (650M) ──── LoRA adapters (rank=8, alpha=16, target: q,v projections)
        │
        │  [CLS] token embedding → classification head
        ▼
Enzyme Family Prediction
        │
        ├── Softmax probabilities (calibrated via temperature scaling)
        ├── Integrated gradients attribution (per-residue importance)
        └── FastAPI inference endpoint
```

---

## Scientific Background

### Enzyme Family Classification

Enzymes are classified hierarchically by the Enzyme Commission (EC) system into six top-level classes (oxidoreductases, transferases, hydrolases, lyases, isomerases, ligases), with further subdivision into subclasses and sub-subclasses. Swiss-Prot provides high-quality, manually reviewed EC annotations for ~570,000 proteins — a gold-standard dataset for supervised learning.

The challenge is generalisation to remote homologues. A model trained on known enzyme families must classify newly discovered sequences that share functional mechanism but have diverged beyond the reach of classical similarity tools. This is where PLM representations offer a principled advantage: they encode evolutionary information across the entire protein universe, not just within training families.

### Why ESM-2

ESM-2 (Lin et al., 2022, *Science*) is a family of protein language models trained on 250 million UniRef90 sequences using a masked language modelling objective. At 650M parameters, ESM-2 produces per-residue and sequence-level representations that capture:

- Local sequence patterns (active site residues, binding motifs)
- Global structural constraints (fold-level organisation)
- Evolutionary co-variation (functional conservation across families)

ESM-2 representations encode structural and functional information without explicit structural supervision, making them a strong foundation for downstream classification tasks.

### Why LoRA

Full fine-tuning of a 650M parameter model requires substantial GPU memory and risks catastrophic forgetting of the evolutionary information encoded during pre-training. Low-Rank Adaptation (LoRA, Hu et al., 2022) addresses both:

- Freezes all pre-trained ESM-2 weights
- Injects trainable low-rank matrices into the query and value projection layers of each attention head
- Updates < 1% of total parameters while achieving performance competitive with full fine-tuning
- Preserves the evolutionary representations that make ESM-2 useful

LoRA rank and alpha are treated as hyperparameters and swept during development (see `config/lora_config.yaml`).

### Why Homology-Aware Splitting

Standard random splits of protein datasets suffer from homology leakage: training and test sequences sharing high sequence identity give inflated performance estimates that do not reflect generalisation to genuinely novel sequences. This project uses MMseqs2 to cluster sequences at 30% identity, ensuring that all sequences in a given cluster appear in only one split. This is the methodologically correct approach for protein ML benchmarks and is increasingly required by journals and competitions.

The performance difference between random and homology-aware splits on this task is expected to be substantial (~15–30% macro-F1 inflation with random splits, based on published benchmarks). Documenting this gap explicitly is one of the scientific contributions of this project.

### Calibration and Uncertainty

A model that is accurate but poorly calibrated — whose confidence scores do not match empirical accuracy — is unreliable in practice. This project applies temperature scaling post-training to calibrate confidence scores, reporting Expected Calibration Error (ECE) and reliability diagrams alongside accuracy metrics. Sequences where the model is uncertain (max softmax probability below a threshold) are flagged rather than silently assigned to the highest-scoring class.

---

## Methodology

### Data Pipeline

Raw data is downloaded from UniProtKB/Swiss-Prot (reviewed entries with EC annotations). The ETL pipeline:

1. Filters to entries with at least one EC number annotation
2. Assigns top-level EC class as the classification label (6 classes), with fine-grained sub-class as a configurable extension
3. Removes sequences shorter than 50 or longer than 1024 residues (ESM-2 context window)
4. Deduplicates at 100% sequence identity
5. Clusters at 30% identity via MMseqs2 for homology-aware splitting
6. Assigns clusters to train (70%) / val (15%) / test (15%) splits

All ETL steps are versioned and logged. Raw and processed data checksums are stored in `data/checksums.json`.

### Fine-Tuning

ESM-2 is fine-tuned with LoRA using the HuggingFace `transformers` and `peft` libraries:

```python
from transformers import EsmForSequenceClassification
from peft import LoraConfig, get_peft_model

model = EsmForSequenceClassification.from_pretrained(
    "facebook/esm2_t33_650M_UR50D",
    num_labels=num_classes
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 2,359,296 || all params: 654,158,850 || trainable%: 0.36
```

Training uses AdamW with cosine learning rate schedule, gradient clipping, and early stopping on validation macro-F1. All hyperparameters are defined in `config/training_config.yaml` and logged to MLflow.

### Calibration

Post-training, temperature scaling is applied on the validation set:

```python
from src.calibration.temperature_scaling import TemperatureScaler

scaler = TemperatureScaler()
scaler.fit(val_logits, val_labels)
calibrated_probs = scaler.calibrate(test_logits)
```

Expected Calibration Error (ECE) is reported before and after calibration. Reliability diagrams are generated for the test set.

### Interpretability — Integrated Gradients

Integrated gradients (Sundararajan et al., 2017) attribute each residue's contribution to the classification decision:

```python
from src.interpretability.integrated_gradients import compute_attributions

attributions = compute_attributions(
    model=model,
    sequence=sequence,
    target_class=predicted_class,
    n_steps=50
)
```

Attribution scores are:

1. Visualised as per-residue importance heatmaps
2. Cross-validated against MSA conservation scores (do high-attribution positions correspond to conserved positions?)
3. Mapped to 3D structure via PDB retrieval and B-factor replacement for visualisation in PyMOL/ChimeraX

---

## Key Design Decisions

**Why LoRA rather than full fine-tuning?**  
Full fine-tuning risks catastrophic forgetting of the evolutionary representations that make ESM-2 useful. LoRA preserves pre-trained weights while adapting the model to the classification task. At rank=8, trainable parameters are 0.36% of total — dramatically lower memory and compute requirements with competitive performance. Rank is ablated (r=4, r=8, r=16) and results reported.

**Why homology-aware splitting rather than random?**  
Random splits inflate performance by ~15–30% on protein classification tasks due to homology leakage. MMseqs2 clustering at 30% identity is the current community standard for fair benchmarking and is required for credible comparison with published results. The performance gap between random and homology-aware splits is explicitly reported as a calibration check.

**Why temperature scaling rather than more complex calibration?**  
Temperature scaling is a single-parameter method that is provably optimal for post-hoc calibration under mild assumptions, computationally trivial, and highly interpretable. More complex methods (isotonic regression, Platt scaling) offer marginal gains at the cost of interpretability.

**Why integrated gradients rather than attention weights?**  
Attention weights are not reliable attributions — they reflect information routing, not feature importance. Integrated gradients are theoretically grounded, satisfy completeness and sensitivity axioms, and produce attributions that can be meaningfully cross-validated against MSA conservation. This cross-validation is the key interpretability result: if high-attribution positions co-localise with conserved positions in multiple sequence alignments, the model is attending to the right biology.

**Why config-driven runs?**  
All training, data, and model hyperparameters are defined in YAML files under `config/`. This enables full reproducibility, systematic hyperparameter sweeps, and MLflow logging without code changes. Every experiment in `experiments/` corresponds to a committed config snapshot.

---

## Evaluation Plan

### Primary Metrics

| Metric | Description |
|---|---|
| Macro-F1 | Primary metric — class-balanced, appropriate for imbalanced EC class distributions |
| Per-class F1 | Identifies which enzyme families are hardest to classify |
| Top-3 accuracy | Relevant for applications where a ranked shortlist is acceptable |
| ECE | Expected Calibration Error — lower is better |
| Reliability diagram | Visual calibration assessment |

### Baselines

All results will be reported against:

- **Zero-shot ESM-2** — frozen embeddings + linear probe, no fine-tuning
- **BLAST similarity** — classical sequence similarity baseline
- **Random forest on ESM-2 embeddings** — non-neural baseline
- **Random split vs. homology-aware split** — quantifying leakage inflation explicitly

### Known Failure Modes to Document

- **Remote homologues at family boundaries**: sequences at the boundary between two enzyme families may be assigned to either class with similar confidence. Calibrated uncertainty flags these cases.
- **Novel enzyme families**: the model cannot classify sequences from families absent in Swiss-Prot. Uncertainty thresholding catches most of these — low max-softmax probability signals out-of-distribution input.
- **Truncated sequences**: sequences near the 1024-residue context window limit may lose C-terminal information. Flagged in inference output.
- **Multi-functional enzymes**: some proteins carry multiple EC annotations. The current implementation uses the primary annotation as the training label — multi-label classification is a planned extension.
- **Homology leakage sensitivity**: if MMseqs2 clustering parameters are set too loosely, some leakage may persist. Sensitivity to clustering threshold is tested explicitly.

---

## Repository Structure

```
esm2-enzyme-classifier/
├── README.md
├── LICENSE
├── pyproject.toml
├── Dockerfile
│
├── config/
│   ├── data_config.yaml           # UniProtKB filters, sequence length limits
│   ├── lora_config.yaml           # LoRA rank, alpha, target modules, dropout
│   ├── training_config.yaml       # LR, batch size, epochs, early stopping
│   ├── calibration_config.yaml    # Temperature scaling settings
│   └── inference_config.yaml      # FastAPI endpoint settings
│
├── src/
│   ├── data/
│   │   ├── download.py            # UniProtKB/Swiss-Prot ETL
│   │   ├── preprocess.py          # Filtering, deduplication, label encoding
│   │   ├── clustering.py          # MMseqs2 wrapper for homology-aware splitting
│   │   └── dataset.py             # PyTorch Dataset class
│   │
│   ├── model/
│   │   ├── esm2_lora.py           # ESM-2 + LoRA model definition
│   │   ├── classifier_head.py     # Classification head
│   │   └── checkpointing.py       # Model saving and loading
│   │
│   ├── training/
│   │   ├── trainer.py             # Training loop
│   │   ├── metrics.py             # Macro-F1, per-class F1, top-k accuracy
│   │   └── callbacks.py           # Early stopping, LR scheduling
│   │
│   ├── calibration/
│   │   ├── temperature_scaling.py # Post-hoc calibration
│   │   └── ece.py                 # Expected Calibration Error + reliability diagrams
│   │
│   ├── interpretability/
│   │   ├── integrated_gradients.py # Per-residue attribution scores
│   │   ├── msa_conservation.py    # MSA conservation score computation
│   │   └── structure_mapping.py   # PDB retrieval + B-factor replacement
│   │
│   ├── inference/
│   │   ├── api.py                 # FastAPI inference endpoint
│   │   └── predict.py             # Batch inference with uncertainty flagging
│   │
│   └── visualisation/
│       ├── attribution_heatmap.py # Per-residue importance visualisation
│       ├── calibration_plots.py   # Reliability diagrams
│       └── confusion_matrix.py    # Per-class performance
│
├── data/
│   ├── raw/                       # Downloaded Swiss-Prot data (not tracked)
│   ├── processed/                 # Filtered, split datasets
│   ├── clusters/                  # MMseqs2 clustering output
│   └── checksums.json             # Data versioning checksums
│
├── notebooks/
│   ├── 01_data_exploration.ipynb          # Dataset statistics, class distribution
│   ├── 02_embedding_analysis.ipynb        # Zero-shot ESM-2 embedding visualisation (UMAP)
│   ├── 03_finetuning_results.ipynb        # Training curves, metric comparison
│   ├── 04_homology_analysis.ipynb         # Homology leakage analysis
│   ├── 05_calibration.ipynb               # ECE, reliability diagrams
│   ├── 06_interpretability.ipynb          # Integrated gradients, MSA cross-validation
│   └── 07_structure_mapping.ipynb         # Attribution → 3D structure visualisation
│
├── experiments/
│   ├── baseline_zeroshot/         # Frozen ESM-2 + linear probe
│   ├── baseline_blast/            # BLAST similarity baseline
│   ├── lora_r4/                   # LoRA rank=4 (ablation)
│   ├── lora_r8/                   # LoRA rank=8 (primary experiment)
│   ├── lora_r16/                  # LoRA rank=16 (ablation)
│   └── full_finetune/             # Full fine-tuning (GPU-intensive ablation)
│
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_clustering.py
│   ├── test_model.py
│   ├── test_calibration.py
│   └── test_api.py
│
└── .github/
    └── workflows/
        └── ci.yml                 # Linting, unit tests, smoke test
```

---

## Quickstart

> **Note:** The project is in active development. Instructions below reflect the intended setup; full reproducibility will be confirmed as each component is completed.

### With Docker (recommended)

```bash
git clone https://github.com/izgys/esm2-enzyme-classifier
cd esm2-enzyme-classifier
docker build -t esm2-enzyme-classifier .
docker run -p 8000:8000 esm2-enzyme-classifier
```

### Local installation

```bash
git clone https://github.com/izgys/esm2-enzyme-classifier
cd esm2-enzyme-classifier
pip install -e ".[dev]"
```

### Data preparation

```bash
python -m src.data.download --config config/data_config.yaml
python -m src.data.preprocess --config config/data_config.yaml
python -m src.data.clustering --config config/data_config.yaml
```

### Training

```bash
python -m src.training.trainer \
    --data-config config/data_config.yaml \
    --lora-config config/lora_config.yaml \
    --training-config config/training_config.yaml \
    --output experiments/lora_r8
```

### MLflow tracking

```bash
mlflow ui
# open http://localhost:5000
```

---

## Roadmap

**v1 — Sequence-based classification**

- [ ] ETL pipeline (UniProtKB/Swiss-Prot download, filtering, label encoding)
- [ ] MMseqs2 homology-aware splitting
- [ ] ESM-2 + LoRA fine-tuning pipeline with MLflow tracking
- [ ] Baseline comparisons (zero-shot ESM-2, BLAST, RF on embeddings)
- [ ] Calibration (temperature scaling, ECE reporting)
- [ ] Integrated gradients interpretability
- [ ] MSA conservation cross-validation
- [ ] FastAPI inference endpoint
- [ ] Docker packaging and CI

**v2 — Structure-aware extension**

- [ ] Attribution mapping to 3D structure (PDB retrieval, B-factor replacement)
- [ ] SaProt integration (structure-aware sequence tokens from Foldseek)
- [ ] Multi-label classification for multi-functional enzymes
- [ ] ESM-3 fine-tuning (sequence + structure + function tokens)

---

## Scientific References

- Lin Z et al. (2022). *Science* — ESM-2: Evolutionary-scale prediction of atomic-level protein structure with a language model. DOI: 10.1126/science.ade2574
- Hu EJ et al. (2022). *ICLR* — LoRA: Low-Rank Adaptation of Large Language Models. arXiv: 2106.09685
- Sundararajan M et al. (2017). *ICML* — Axiomatic Attribution for Deep Networks (Integrated Gradients). arXiv: 1703.01365
- Guo C et al. (2017). *ICML* — On Calibration of Modern Neural Networks (Temperature Scaling). arXiv: 1706.04599
- Steinegger M & Söding J (2017). *Nature Methods* — MMseqs2 enables sensitive protein sequence searching. DOI: 10.1038/nmeth.4176
- The UniProt Consortium (2023). *Nucleic Acids Res* — UniProt: the Universal Protein Knowledgebase. DOI: 10.1093/nar/gkac1052
- Rao R et al. (2021). *PNAS* — MSA Transformer. DOI: 10.1073/pnas.2016239118
- Su J et al. (2023). *bioRxiv* — SaProt: Protein Language Modeling with Structure-Aware Vocabulary. DOI: 10.1101/2023.10.01.560349

---

## Author

**Iker Zapirain Gysling**  
Computational Biochemist, PhD  
Barcelona, Spain  
[LinkedIn](https://linkedin.com/in/zgysling) · [GitHub](https://github.com/izgys)

---

## License

MIT License — see `LICENSE` for details.
