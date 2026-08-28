# PB-NCO

Official implementation of **“Enabling Population-Based Architectures for Neural Combinatorial Optimization.”** PB-NCO solves Maximum Cut (MC) and Maximum Independent Set (MIS) using contextual neural improvement (cNI), conditioned neural construction (cNC), or both.

## Setup

Python 3.12 and a CUDA-capable GPU are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Evaluation instances are distributed separately and are not included in this repository. Download the dataset archive and extract it into the repository root so that the folders appear as `data/ER20/`, `data/ER100/`, and so on.

**Before publishing:** replace this sentence with the permanent dataset URL and DOI. Zenodo is recommended for the archival copy.

## Evaluate pretrained models

Maximum Cut:

```bash
python eval.py \
  --problem mc \
  --ni_model_load_path checkpoints/mc_ni/cNI_mc.pth \
  --nc_model_load_path checkpoints/mc_nc/cNC_mc_ER.pth \
  --eval_graph_types ER700_800 \
  --num_eval_graphs 128 \
  --max_time_per_instance 60 \
  --save_results
```

Maximum Independent Set:

```bash
python eval.py \
  --problem mis \
  --ni_model_load_path checkpoints/mis_ni/cNI_mis.pth \
  --nc_model_load_path checkpoints/mis_nc/cNC_mis_ER.pth \
  --eval_graph_types ER700_800 \
  --num_eval_graphs 128 \
  --max_time_per_instance 60 \
  --save_results
```

For RB instances, replace the cNC checkpoint with `cNC_mc_RB.pth` or `cNC_mis_RB.pth`. Use `python eval.py --help` for all evaluation options.

## Train

```bash
# cNI
python ni_train.py --problem mc --save_models

# cNC
python nc_train.py --problem mc --nc_train_mode conditioned_network --save_model
```

Use `--problem mis` for MIS and `--debug` for a short smoke run. Weights & Biases is optional: install it with `pip install wandb` and add `--wandb`.

## Contents

```text
checkpoints/  Six pretrained MC/MIS checkpoints
args/         Command-line configuration
env/          MC/MIS environments
nets/         Neural network models
utils/        Loading and utility functions
eval.py       Evaluation entry point
nc_train.py   cNC training entry point
ni_train.py   cNI training entry point
```

## License and citation

Released under the [MIT License](LICENSE). Please cite the accompanying paper; the final BibTeX entry will be added after publication.
