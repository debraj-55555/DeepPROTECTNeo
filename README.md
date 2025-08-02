# DeepPROTECTNeo: A Deep learning-based Personalized and RV-guided Optimization tool for TCR Epitope interaction using Context-aware Transformers

[BioRxiv](https://doi.org/10.1101/2025.01.04.631301) [Zenodo](https://dummy.org/zenodo)

A high-performance, explainable deep learning model for Neo-epitope prioritizationm leveraging the prediction of TCR–epitope binding, integrating rotary embeddings, explicit cross-attention, and integrated gradients.

---

## Contents
1. [Installation](#installation)
2. [Usage](#usage)
   - [Training on Custom Data](#training-on-custom-data)
   - [Validation Split Logic](#validation-split-logic)
   - [Running Tests](#running-tests)
3. [Repository Structure](#repository-structure)
4. [Provided Weights](#provided-weights)
5. [Citation](#citation)

---

## Installation

We recommend using **Conda**. An `env.yml` file is provided to recreate the exact environment.

```bash
conda env create -f env.yml
conda activate deepprotectneo
```

Install anarci (see [ANARCI installation guidelines](https://github.com/oxpig/ANARCI?tab=readme-ov-file#installation)) in the same environment for code execution.

---

## Usage

### Training on Custom Data

Place your **training** (and optional **validation**) CSV files in the `data/` directory. Each file **must** have three columns (with header):

```csv
TCR,epitope,Label
CASSLGNEQF,EAAGIGILTV,1
CASSLGVATGELF,EAAGIGILTV,1
...
```

- `--train_file`: filename under `data/` for training set (CSV).
- `--val_file` (optional): filename under `data/` for a separate validation set. If omitted, an 80/20 train/validation split is performed automatically by TCR grouping.
- **Batch size**, **epochs**, **device**, and **early stopping** flags are available.
- All outputs (models, logs, plots) are saved under `run/{save_dir}/`.

**Example**:
```bash
python main.py   --mode train   --train_file example_train.csv   --val_file example_val.csv   --batch_size 128   --epochs 30   --device cuda:0   --early_stopping   --run_dir run   --save_dir exp1
```

### Validation Split Logic
- If `--val_file` is provided, the script uses that CSV directly.  
- If not, it splits `--train_file` randomly into 80% train / 20% validation, **grouped by TCR** to avoid data leakage.

### Running Tests

After training, your best and early-stop weights are saved in `run/{save_dir}/best_model.pth` and `early_stop_model.pth`.

To evaluate on a test CSV (same format as above):
```bash
python main.py   --mode test   --test_file data/test_dataset.csv   --model_path run/exp1/best_model.pth   --predictions_file run/exp1/predictions.csv   --metric_file run/exp1/metrics.json   --batch_size 128   --device cuda:0
```

This produces:  
- `predictions.csv`: all TCR–epitope pairs, binding scores, predicted labels.  
- `metrics.json`: evaluation metrics (accuracy, F1, AUROC, AUPRC, etc.).

---

## Repository Structure

```
├── cache/               # ANARCI cache directory
├── data/                # Place train/val/test CSVs here
├── models/              # Model architecture code
├── dataloader/          # Dataset & tokenizer modules
├── scripts/             # Utility scripts (e.g., main_utils)
├── run/                 # Default output directory
├── weights/             # Provided 5-fold pretrained weights
├── env.yml              # Conda environment specification
├── params.json          # Hyperparameter configuration
├── main.py              # Entry point (train & test)
└── README.md            # This file
```

---

## Custom Testing with Pre-trained

Pre-trained 5-fold cross-validation weights are available via [this link](https://drive.google.com/drive/folders/1Bgc9CdsYmt0TWE2KM5vAYEKpPq7tvK-G?usp=sharing). Download them and put them inside ```weights/``` folder. And the test command can be run using the following argument 

```bash
--model_path weights/fold_{i}_model.pth
```

---

## Citation

Please cite our preprint:

```bibtex
@article {Das2025.01.04.631301,
	author = {Das, Debraj and Bhaduri, Soumyadeep and Pramanick, Avik and Mitra, Pralay},
	title = {DeepPROTECTNeo: A Deep learning-based Personalized and RV-guided Optimization tool for TCR Epitope interaction using Context-aware Transformers},
	elocation-id = {2025.01.04.631301},
	year = {2025},
	doi = {10.1101/2025.01.04.631301},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/01/05/2025.01.04.631301},
	eprint = {https://www.biorxiv.org/content/early/2025/01/05/2025.01.04.631301.full.pdf},
	journal = {bioRxiv}
}
```

**Zenodo**: https://doi.org/10.5281/zenodo.xxxxxxx
