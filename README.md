# Metagenome-AI

A comprehensive framework for protein sequence classification using Large Protein Language Models (pLMs) with support for multiple embedding models and classifier architectures.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Training](#basic-training)
  - [Fine-tuning Protein Language Models](#fine-tuning-protein-language-models)
  - [Program Modes](#program-modes)
- [Configuration](#configuration)
- [Dataset Format](#dataset-format)
- [Notebooks](#notebooks)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Overview

Metagenome-AI is a flexible framework for protein sequence classification that leverages state-of-the-art protein language models to generate embeddings and trains lightweight classifiers for various downstream tasks. The framework supports multiple embedding models (ESM-2, ESM3, ProtTrans) and classifier architectures (MLP, XGBoost), making it adaptable to diverse protein classification problems including antimicrobial peptide (AMP) prediction, toxicity prediction, and Gram-positive/negative activity classification.

## Features

- **Multiple Embedding Models**: Support for ESM-2 (650M, 3B parameters), ESM3, ProtTrans, and ProteinVec
- **Flexible Classifiers**: Choose between MLP and XGBoost classifiers
- **Fine-tuning Support**: Fine-tune protein language models on custom datasets
- **Multi-GPU Training**: Distributed training support using PyTorch DDP
- **Modular Architecture**: Easy to extend with new models and classifiers
- **Experiment Tracking**: Integration with Weights & Biases (WandB)
- **Memory Efficient**: Separate embedding generation and classifier training phases

## Architecture

The framework follows a two-stage pipeline:

1. **Embedding Generation**: Protein sequences are processed by pre-trained language models to generate fixed-size embeddings
2. **Classifier Training**: Lightweight classifiers are trained on the generated embeddings for specific classification tasks

This approach allows for efficient experimentation with different classifiers without re-computing embeddings.

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- Conda or pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Metagenome-AI.git
cd Metagenome-AI
```

2. Create and activate a conda environment:
```bash
conda create -n mai python==3.9
conda activate mai
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) For ESM3 model access, configure Hugging Face authentication:
```bash
huggingface-cli login
```

## Usage

### Basic Training

To run the complete pipeline (embedding generation + classifier training):

```bash
python src/train.py -c src/configs/config_sample.json
```

### Fine-tuning Protein Language Models

To fine-tune a protein language model on your dataset:

```bash
python src/finetune.py -c src/configs/config_ft.json
```

Or using Hugging Face's training framework:

```bash
python src/finetuning_hf.py -c src/configs/config_ft.json
```

### Program Modes

The framework supports three operational modes:

1. **RUN_ALL** (default): Generate embeddings and train classifier
```bash
python src/train.py -c src/configs/config_sample.json
```

2. **ONLY_STORE_EMBEDDINGS**: Only generate and store embeddings
```json
{
  "program_mode": "ONLY_STORE_EMBEDDINGS",
  ...
}
```

3. **TRAIN_PREDICT_FROM_STORED**: Train classifier using pre-computed embeddings
```json
{
  "program_mode": "TRAIN_PREDICT_FROM_STORED",
  ...
}
```

### Example Configurations

The `src/configs/` directory contains several example configurations:

- `config_sample.json`: Basic ESM-2 with XGBoost classifier
- `config_esm2.json`: ESM-2 3B model configuration
- `config_esm3.json`: ESM3 model configuration
- `config_protein_trans.json`: ProtTrans model configuration
- `config_ft.json`: Fine-tuning configuration

## Configuration

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `model_type` | Embedding model type | `"ESM"`, `"ESM3"`, `"PTRANS"`, `"PVEC"` |
| `classifier_type` | Classifier architecture | `"MLP"`, `"XGBoost"` |
| `train` | Path to training dataset | `"data/sample_train.tsv"` |
| `valid` | Path to validation dataset | `"data/sample_validation.tsv"` |
| `test` | Path to test dataset | `"data/sample_test.tsv"` |
| `emb_dir` | Directory for storing embeddings | `"emb_dir/sample"` |
| `model_folder` | Directory for model checkpoints | `"classifier_results/sample"` |
| `model_basename` | Basename for saved models | `"esm_sample"` |
| `wandb_key` | Weights & Biases API key | Your WandB API key |

### Optional Parameters

#### General Parameters
- `program_mode`: Execution mode (default: `"RUN_ALL"`)
- `batch_size`: Batch size for training (default: `32`)
- `max_tokens`: Maximum tokens per batch for embedding generation (default: `2500`)
- `log_dir`: Directory for log files (default: `"./logs/"`)
- `pred_dir`: Directory for predictions (default: `"./predictions/"`)
- `classifier_path`: Path to pre-trained classifier to skip training

#### MLP Classifier Parameters
- `num_epochs`: Number of training epochs (default: `10`)
- `lr`: Learning rate (default: `0.001`)
- `hidden_layers`: Hidden layer sizes, e.g., `[1024, 512]`
- `early_stop_patience`: Early stopping patience (default: `4`)

#### XGBoost Classifier Parameters
- `objective`: Objective function (default: `"multi:softmax"`)
- `n_estimators`: Number of trees (default: `10`)
- `eta`: Learning rate (default: `0.001`)
- `early_stop`: Early stopping rounds (default: `4`)
- `max_depth`: Maximum tree depth (default: `8`)
- `eval_metric`: Evaluation metric (default: `"mlogloss"`)
- `verbosity`: Verbosity level (default: `1`)

#### Fine-tuning Parameters
- `num_epochs_finetune`: Number of fine-tuning epochs
- `batch_size_finetune`: Batch size for fine-tuning
- `max_mask_prob`: Maximum masking probability for MLM training
- `model_name_or_path`: Path to base model

## Dataset Format

Datasets should be in TSV (tab-separated values) format with the following columns:

```
<protein_id>    <length>    <sequence>    <label>
```

Example:
```
protein_001    150    MKTIIALSYIFCLVFA...    1
protein_002    89     ARTKQTARKSTGGKA...     0
```

For multi-label classification, additional label columns can be added:
```
<protein_id>    <length>    <sequence>    <label1>    <label2>    <label3>
```

Sample datasets are provided in the `data/` directory.

## Notebooks

The `notebooks/` directory contains Jupyter notebooks for data analysis and experimentation:

### Needleman-Wunsch Similarity Analysis (`Needleman-Wunsch.ipynb`)

This notebook performs sequence similarity analysis between predicted antimicrobial peptides (from RiPP core peptides) and known AMP databases. The analysis uses the Needleman-Wunsch global alignment algorithm to calculate pairwise similarity scores between query sequences and a reference database. The notebook processes over 16,000 RiPP core peptide sequences, computes their alignment scores against known AMPs, normalizes the scores by sequence length to obtain percentage identity, and visualizes the distribution of similarity scores through publication-quality histograms. This analysis helps assess the novelty of predicted antimicrobial candidates by quantifying their sequence similarity to previously characterized AMPs. The notebook also supports comparison with DIAMOND BLASTP for computational efficiency, though the primary focus is on the more sensitive Needleman-Wunsch approach.

### Model Performance Analysis (`amp_tox_ripp-round2.ipynb`)

This comprehensive notebook evaluates and compares multiple protein language models for antimicrobial peptide prediction across four classification tasks: global AMP activity, Gram-positive activity, Gram-negative activity, and toxicity prediction. The analysis includes performance benchmarking of ESM2-650M, ESM2-3B, ESM3, and ProtTrans models against existing AMP prediction tools (Macrel, AMPScanner, iAMPpred, amPEPpy, and ToxinPred3) using metrics including accuracy, F1-score, AUC, and MCC. The notebook processes predictions from approximately 16,700 RiPP core peptides derived from metagenomics and microbial genomes, applies ensemble prediction strategies by intersecting predictions from multiple models, and filters candidates based on normalized probability scores to identify high-confidence antimicrobial peptides with low toxicity. Advanced visualizations include Venn diagrams showing model prediction overlaps and comparative bar plots demonstrating that the ESM2-3B model achieves superior performance (>90% accuracy) compared to traditional AMP prediction tools.

## Project Structure

```
Metagenome-AI/
├── data/                          # Sample datasets and predictions
│   ├── sample_train.tsv
│   ├── sample_validation.tsv
│   ├── sample_test.tsv
│   └── predictions/               # Model predictions output
├── src/                           # Source code
│   ├── train.py                   # Main training script
│   ├── finetune.py                # Model fine-tuning script
│   ├── config.py                  # Configuration management
│   ├── dataset.py                 # Dataset classes
│   ├── configs/                   # Configuration files
│   ├── embeddings/                # Embedding model implementations
│   │   ├── embedding_esm.py
│   │   ├── embedding_esm3.py
│   │   ├── embedding_protein_trans.py
│   │   └── embedding_protein_vec.py
│   ├── classifiers/               # Classifier implementations
│   │   ├── classifier_mlp.py
│   │   └── classifier_xgboost.py
│   ├── finetuning/                # Fine-tuning utilities
│   └── utils/                     # Utility functions
│       ├── metrics.py
│       ├── wandb.py
│       └── early_stopper.py
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── Needleman-Wunsch.ipynb
│   └── amp_tox_ripp-round2.ipynb
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, suggestions, or issues:

- **Create an issue**: [GitHub Issues](https://github.com/BGIResearch/Metagenome-AI/issues)
- **Email**: vladimir.kovacevic@etf.rs

---

**Developed at**: BGI Research

**Contributors**: Nikola Milicevic, Vladimir Kovacevic
