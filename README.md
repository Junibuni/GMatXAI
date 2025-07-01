# GMatXAI: GNN-Based Materials Property Prediction

GMatXAI is a deep learning framework for predicting a scalar material property—specifically, **formation energy per atom**—using a custom hybrid Graph Neural Network (GNN) architecture. It supports data from Materials Project, JARVIS, or a combination of both, and uses PyTorch Geometric as its graph learning backend.

---

## Key Features

- **Predicts formation energy per atom** from atomic crystal structures
- Combines **local (CartNet)** and **global (Matformer)** GNN modules
- Supports **Materials Project**, **JARVIS**, and **combined** datasets
- Configurable training using YAML files
- Built-in attention fusion mechanism
- Compatible with CUDA-enabled GPUs (tested with PyTorch 2.6.0 + cu118)

---

## Model Architecture: UniCrystalFormer

The core model is `UniCrystalFormer`, which integrates two GNN backbones:

### -  AtomEncoder
- Embeds atomic numbers and external MEGNet features
- Uses a **gated fusion** mechanism to combine embeddings

### -  Local Encoder: CartNetBlock
- Captures **short-range interactions**
- Applies multiple layers of `CartNet_layer` with radius-based neighbor cutoff

### -  Global Encoder: MatformerBlock
- Captures **long-range dependencies** using MatformerConv (Transformer-based GNN)
- Applies edge-augmented attention layers with MLP + GraphNorm

### -  Fusion
- Local and global embeddings are combined at each layer:
  - By **sum**, or
  - Using an **attention-based fusion mixer**

### -  Readout & Regression
- Uses `Set2Set` pooling for graph-level embedding
- Outputs a scalar prediction via a feed-forward head

---

## Installation

### Recommended Python version: `3.10.10`

### Windows (auto setup)

```bash
setup.bat
```

### Manual setup

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu118.html
pip install -r requirements.txt
```

## Dataset Preparation

This project provides a dataset generation script that collects atomic structure data from the **Materials Project (MP)** and **JARVIS (JV)** databases. These are automatically preprocessed and saved in `train.pkl`, `val.pkl`, and `test.pkl` formats.

### Run Dataset Generator

Use the following command to create datasets:

```bash
make_dataset.bat
```
This will internally call make_dataset.py, which supports the following datasets:

- "mp": Materials Project only
- "jv": JARVIS only
- "mpjv": Concatenation of MP and JV (default)
- "jvmp": Same as "mpjv" but loading JV first (equivalent in content)

You may specify the dataset type and number of entries like this:

```bash
python make_dataset.py --dataset mpjv --num_entries 1000 --target formation_energy_per_atom band_gap
```

### What Happens Internally
1. Materials Project Data (via MAPI):

   -  Uses MPRester to fetch structures with specific scalar properties.

    - Fields like "formation_energy_per_atom" and "band_gap" are retrieved.

    - Converts MP structures to JARVIS-style dictionary format using mp_to_jarvis().

2. JARVIS Data (via Figshare):

    - Downloads from JARVIS DFT dataset.

    - Maps fields like formation_energy_peratom → formation_energy_per_atom.

3. Merge and Split:

    - The combined dataset is randomly shuffled.

    - Split into:

        - 80% training

        - 10% validation

        - 10% test

    - Saved as:

        - train.pkl

        - val.pkl

        - test.pkl

These files are saved in the `data/<dataset>/` directory.

### Target Properties
By default, the script extracts the following scalar properties:

- formation_energy_per_atom (default)

- band_gap

- energy_above_hull

You can customize the list of target properties using the `--target` flag.

### Tip
Make sure you have set your Materials Project API key (MAPI) in a .env file like this:

```ini
MAPI=your_api_key_here
```
This is required to access MP data.

## Model Training
Run training using a config file:

```bash
python main.py --config configs/test.yaml
```
You may adjust hyperparameters and dataset paths in the YAML file.

## Project Structure
```bash
GMatXAI/
├── main.py                  # Training entry point
├── make_dataset.bat         # Generate datasets
├── setup.bat                # Auto environment setup
├── requirements.txt         # Dependency list
├── configs/                 # Training configuration files
├── src/
│   ├── data/                # Dataset and preprocessing scripts
│   ├── model/               # GNN model architecture
│   ├── train/               # Training pipeline
│   ├── utils/               # Utilities (logging, parsing, etc.)
├── outputs/                 # Saved models, logs, predictions
└── README.md
```

## Dependencies
- Python ≥ 3.10

- PyTorch 2.6.0

- PyTorch Geometric

- NumPy, pandas, scikit-learn

- tqdm, matplotlib, pyyaml

## Output
All training artifacts (logs, checkpoints, results) are saved under:
``` bash
outputs/
```