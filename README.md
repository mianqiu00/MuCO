# MuCO: Generative Peptide Cyclization Empowered by Multi-stage Conformation Optimization

MuCO is a deep learning framework for generating 3D conformations of cyclic peptides through a multi-stage pipeline. The model decomposes the complex task of cyclic peptide structure prediction into three sequential stages: backbone generation, sidechain packing, and force field relaxation with cyclization validation.

## Overview

Cyclic peptides are promising therapeutic candidates due to their enhanced stability, bioavailability, and target specificity. However, predicting their 3D structures remains challenging due to the conformational constraints imposed by cyclization. MuCO addresses this challenge through a hierarchical generation approach:

1. **Stage 1: Backbone Generation** - Generates the protein backbone (N, CA, C, O atoms) using SE(3) flow matching
2. **Stage 2: Sidechain Generation** - Predicts sidechain conformations conditioned on the generated backbone using equivariant neural networks
3. **Stage 3: Relaxation & Validation** - Optimizes the full-atom structure using molecular mechanics force fields and validates cyclization

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.x or later (for GPU acceleration)
- Conda (recommended)

### Setup Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate muco

# Install additional dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install lightning omegaconf einops biopython
pip install fair-esm  # For ESM2 sequence encoder
```

## Project Structure

```
MuCO/
├── backbone_train.py          # Entry point for backbone model training
├── backbone_sample.py         # Entry point for backbone sampling/inference
├── sidechain_train.py         # Entry point for sidechain model training
├── sidechain_sample.py        # Entry point for sidechain sampling/inference
├── config/
│   ├── backbone_train.yaml    # Backbone training configuration
│   ├── sidechain_train.yaml   # Sidechain training configuration
│   └── sidechain_sample.yaml  # Sidechain sampling configuration
├── model/
│   ├── backbone/              # Backbone generation model (FoldFlow2-based)
│   │   ├── flow_model.py      # Main model architecture
│   │   ├── structure_network.py  # IPA-based encoder/decoder
│   │   ├── trunk.py           # Triangular attention transformer
│   │   ├── flow/              # SE(3) flow matching modules
│   │   └── components/        # Network components (ESM2, IPA, etc.)
│   └── sidechain/             # Sidechain generation model
│       ├── models/
│       │   ├── cnf.py         # Continuous Normalizing Flow wrapper
│       │   └── equiformer_v2/ # EquiformerV2 architecture
│       └── utils/             # Training and evaluation utilities
├── runner/
│   ├── backbone_trainer.py    # Backbone training logic
│   ├── backbone_sampler.py    # Backbone inference logic
│   ├── sidechain_trainer.py   # Sidechain training logic
│   └── sidechain_sampler.py   # Sidechain inference logic
├── relaxer/                   # Force field relaxation module
│   ├── auto.py                # Automatic cyclization mode detection
│   ├── relax.py               # Batch relaxation pipeline
│   ├── head_tail.py           # Head-to-tail cyclization
│   ├── cys_to_cys.py          # Disulfide bond cyclization
│   ├── k_to_de.py             # Isopeptide bond cyclization (Lys-Asp/Glu)
│   ├── analysis_energy.py     # Energy analysis and visualization
│   └── analysis_structure.py  # Structural analysis
├── openfold/                  # OpenFold utilities
├── data/
│   ├── metadata.csv           # Dataset metadata
│   └── pdb_reader.py          # PDB parsing utilities
└── environment.yml            # Conda environment specification
```

## Pipeline

### Stage 1: Backbone Generation

The backbone generation model is based on **FoldFlow2** architecture with SE(3) flow matching. It generates backbone atom coordinates (N, CA, C, O) conditioned on amino acid sequences.

**Key Features:**
- ESM2 (650M) sequence encoder for rich residue representations
- Invariant Point Attention (IPA) for geometric reasoning
- SE(3) flow matching for joint translation and rotation generation
- Triangular self-attention for capturing pairwise residue interactions

**Training:**
```bash
python backbone_train.py
```

Configuration is loaded from `config/backbone_train.yaml`. Key parameters:
- `data.csv_path`: Path to training data metadata
- `experiment.batch_size`: Training batch size
- `experiment.num_epoch`: Number of training epochs
- `experiment.num_gpus`: Number of GPUs for distributed training

**Sampling:**
```bash
python backbone_sample.py \
    --config_timestamp {timestamp} \
    --ckpt_epoch 100 \
    --output ./data/inference/pdb/coarse \
    --device 0 \
    --batch_size 1024 \
    --split CPBind
```

### Stage 2: Sidechain Generation

The sidechain model predicts torsion angles (chi angles) for all amino acid sidechains given the backbone structure. It uses **EquiformerV2** as the backbone network within a Continuous Normalizing Flow (CNF) framework.

**Key Features:**
- EquiformerV2 with SO(2) equivariant attention for geometric feature learning
- CNF-based generation for smooth and diverse sidechain conformations
- Torsion angle representation on SO(2) manifold
- Optional confidence model for sample quality estimation

**Training:**
```bash
python sidechain_train.py config/sidechain_train.yaml \
    --devices 4 \
    --strategy auto \
    --precision 32-true
```

To resume training from a checkpoint:
```bash
python sidechain_train.py config/sidechain_train.yaml --resume
```

**Sampling:**
```bash
python sidechain_sample.py config/sidechain_sample.yaml output_name \
    --seed 42 \
    --save_traj False
```

### Stage 3: Force Field Relaxation

The relaxation stage uses OpenMM to perform energy minimization and validate successful cyclization. It automatically detects the cyclization mode based on terminal residue chemistry.

**Supported Cyclization Modes:**
- **Head-to-tail**: Standard backbone cyclization (N-terminal N to C-terminal C)
- **Cys-to-Cys**: Disulfide bond formation (SG-SG)
- **Isopeptide**: Lys-Asp/Glu sidechain cyclization (NZ to CG/CD)

**Single Structure Relaxation:**
```bash
cd relaxer
python auto.py input.pdb output.pdb
```

**Batch Relaxation:**
```bash
cd relaxer
python relax.py
```

Configure input/output paths and GPU settings in `relax.py`:
```python
INPUT_ROOT = "./unrelaxed"
OUTPUT_ROOT = "./relaxed"
GPU_IDS = [0, 1, 2, 3]
WORKERS_PER_GPU = 4
```

**Analysis:**
```bash
# Energy analysis and visualization
python analysis_energy.py

# Structural analysis
python analysis_structure.py
```

Output includes:
- Per-structure energy before/after relaxation
- Cyclization success rate
- Bond distance validation
- Statistical comparisons across datasets

## Configuration

### Backbone Training (`config/backbone_train.yaml`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data.csv_path` | Path to dataset metadata CSV | - |
| `data.filtering.max_len` | Maximum sequence length | 512 |
| `experiment.batch_size` | Training batch size | 256 |
| `experiment.num_epoch` | Number of training epochs | 100 |
| `experiment.learning_rate` | Learning rate | 2e-4 |
| `flow_matcher.ot_plan` | Use optimal transport pairing | False |
| `model.esm2_model_key` | ESM2 model variant | esm2_650M |

### Sidechain Training (`config/sidechain_train.yaml`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.hidden_dim` | Hidden dimension | 128 |
| `model.num_layers` | Number of EquiformerV2 layers | 4 |
| `training.batch_size` | Training batch size | 32 |
| `training.lr` | Learning rate | 1e-4 |
| `training.epochs` | Number of training epochs | 100 |

## Data Format

Training data should be organized as PDB files with metadata in a CSV file composed by data/pdb_reader.py

## Evaluation Metrics

### Backbone Quality
- **RMSD**: Root Mean Square Deviation from reference structures
- **Distance Matrix Error**: Pairwise Cα distance deviation

### Sidechain Quality
- **Chi Angle MAE**: Mean Absolute Error for torsion angles (χ1, χ2, χ3, χ4)
- **Sidechain RMSD**: All-atom sidechain RMSD

### Cyclization Validation
- **Success Rate**: Percentage of structures with valid cyclization bonds
- **Final Energy**: Force field energy after minimization (kJ/mol)
- **Bond Distance**: Distance between cyclization atoms (Å)

## Hardware Requirements

- **Training**: 4-8 NVIDIA GPUs with 24GB+ memory (RTX4090 recommended)
- **Inference**: Single GPU with 16GB+ memory
- **Relaxation**: CPU or GPU (OpenMM with CUDA support)

## License

This project is for academic research purposes.

## Acknowledgments

This project builds upon several excellent open-source projects:
- [FoldFlow](https://github.com/DreamFold/FoldFlow) - SE(3) flow matching for protein structure
- [EquiformerV2](https://github.com/atomicarchitects/equiformer_v2) - Equivariant transformer
- [OpenFold](https://github.com/aqlaboratory/openfold) - Protein structure prediction
- [ESM](https://github.com/facebookresearch/esm) - Protein language models
- [OpenMM](https://openmm.org/) - Molecular dynamics simulation
