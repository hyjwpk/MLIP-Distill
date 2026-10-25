# MLIP-Distill

[Architecture-agnostic knowledge distillation for machine-learning interatomic potentials](https://doi.org/10.1016/j.neunet.2026.109742)\
*Neural Networks*, 2027, **206**, 109742.

This repository implements a **knowledge distillation** method for machine-learning interatomic potentials. The method trains a student potential by combining reference energy and force supervision with **atomic-energy** alignment and **energy-Hessian** supervision from a teacher. These complementary signals transfer per-atom energy information and the local force response to atomic displacements without requiring matching hidden representations.

## Setup

The provided environment uses PyTorch with CUDA 11.8 support.

### 1. Clone the repository

```bash
git clone https://github.com/hyjwpk/MLIP-Distill.git
cd MLIP-Distill
```

### 2. Create and activate the environment

```bash
conda env create -f env.yml
conda activate distill
```

This environment includes the modified FAIR-Chem package for student training and eSEN teacher-label generation.

### 3. Install MACE (optional)

```bash
python -m pip install "numpy>=1.26,<2" \
  "mace-torch @ git+https://github.com/hyjwpk/MLIP-Distill.git@deps.mace"
```

This step is required only for MACE teacher-label generation and evaluation.

Run all commands below from the repository root.

## Using the method

The workflow is: prepare structure data, choose a student and distillation objective, generate teacher labels, and train the student.

### Model requirements

For Joint distillation, the teacher must expose per-atom energies and the coordinate derivatives needed to generate Hessian labels. The student must expose compatible per-atom energies and support the derivatives required by the selected Hessian loss implementation.

Student models use FAIR-Chem's model interface. The distillation trainer expects `energy`, `forces`, and, when atomic-energy alignment is enabled, `node_energy` outputs.

Teacher predictions are precomputed and stored as labels, which are loaded during student training.

### Prepare your data

Prepare your data in a format readable by FAIR-Chem's `LmdbDataset`, with atomic numbers, positions, reference energies and forces, and cell information where applicable. Configure field mappings, periodicity, neighbor settings, normalization, and element-energy references for your system.

A suggested layout is:

```text
experiments/resources/data/my-dataset/
├── train/
├── val/
├── test/
└── labels/
    └── my-teacher/
        ├── node_energy/
        ├── force_jacobians/
        └── val_forces/
```

Paths are configurable. Each label directory contains LMDB files. Label keys are the sequential indices of structures in the corresponding split, starting at zero. Values are stored as float32 arrays. Hessian labels are stored as force Jacobians, which equal the negative energy Hessians when forces are derived from energy. Labels must match the structure order, atom order, and units of the dataset.

### Configure the student and objective

Start from a configuration for the student architecture you want to use. For example:

```bash
cp experiments/runs/accuracy/configs/SPICE/MACE-OFF/solvated/joint/painn.yml my-distillation.yml
```

Edit the copied configuration and its included settings for your application:

| Setting                                                          | Purpose                                                    |
| ---------------------------------------------------------------- | ---------------------------------------------------------- |
| `dataset.train.src`, `dataset.val.src`, `dataset.test.src` | Structure-data paths                                       |
| `dataset.train.teacher_labels_folder`                          | Teacher-label root                                         |
| `dataset.train.transforms`                                     | Normalization and element-energy references                |
| `model`                                                        | Student architecture, size, cutoff, and neighbor settings  |
| `outputs`                                                      | Energy, force, and atomic-energy outputs                   |
| `loss_functions`                                               | Reference-label and distillation losses                    |
| `optim`                                                        | Optimizer, training budget, batching, and Hessian sampling |
| `evaluation_metrics`                                           | Metrics and model-selection criterion                      |

Set element-energy references and scaling parameters for your data and model. Shared included files apply to every configuration that includes them.

Use `trainer: src.distill_trainer.DistillTrainer` for distillation. The two distillation terms are controlled by the coefficients of `hessian` and `node_energy`:

| Objective     | `hessian` coefficient | `node_energy` coefficient | Required teacher labels                                                      |
| ------------- | -------------------------- | --------------------------- | ---------------------------------------------------------------------------- |
| Atomic Energy | Zero                       | Positive                    | Teacher atomic energies on the training set                                  |
| Hessian       | Positive                   | Zero                        | Hessian labels on the training set and teacher forces on the validation set |
| Joint         | Positive                   | Positive                    | All three label types                                                        |

Keep `hessian` followed by `node_energy` as the final two entries in `loss_functions`, including when a coefficient is zero. Retain the reference energy/force loss terms required by your task. For training without distillation, use the supplied `no-distillation` configurations with `trainer: ocp`.

When Hessian supervision is enabled, the trainer uses validation teacher-force errors to adjust its weight.

## Generate teacher labels

### MACE-OFF and MACE-MP

To generate the labels needed for Joint distillation with a MACE-OFF teacher:

```bash
python -m experiments.scripts.labels.mace_off \
  --dataset experiments/resources/data/my-dataset \
  --model /path/to/MACE-OFF23_large.model \
  --output experiments/resources/data/my-dataset/labels/MACE-OFF \
  --labels node-energy hessian val-forces
```

For MACE-MP, use `experiments.scripts.labels.mace_mp` with a periodic dataset and a MACE-MP checkpoint.

Both scripts accept one or more label types:

| Option           | Output                                             | Split      |
| ---------------- | -------------------------------------------------- | ---------- |
| `node-energy`  | Atomic-energy contributions                        | Train      |
| `hessian`      | Hessian labels                                    | Train      |
| `train-forces` | Teacher forces                                     | Train      |
| `val-forces`   | Teacher forces                                     | Validation |

Only the selected label types are generated. Existing selected outputs are not overwritten, so `--labels node-energy` can add atomic energies when Hessian and validation-force labels are already available.

### eSEN-OAM

eSEN labels are generated through a separate FAIR-Chem entry point:

```bash
python -m experiments.scripts.labels.esen \
  --config-yml my-distillation.yml \
  --dataset.train.teacher_checkpoint_path=/path/to/esen-checkpoint.pt \
  --dataset.train.teacher_labels_folder=experiments/resources/data/my-dataset/labels/eSEN-OAM \
  --dataset.train.teacher_weights=checkpoint \
  --dataset.train.label_force_batch_size=1 \
  --dataset.train.label_jac_batch_size=1 \
  --dataset.train.vectorize_teach_jacs=False
```

This generates training atomic energies, forces, and Hessian labels, plus validation forces. Use an output location without existing label directories.

## Train the student

After preparing the data and labels:

```bash
python -m src.main --mode train \
  --config-yml my-distillation.yml \
  --run-dir experiments/runs/my-student \
  --seed 0
```

Choose a separate output directory for each run.

## Use the trained student

Set `CHECKPOINT` to a saved student checkpoint and provide a test dataset:

```bash
CHECKPOINT=/path/to/checkpoint.pt

python -m src.main --mode predict \
  --config-yml my-distillation.yml \
  --checkpoint "$CHECKPOINT" \
  --dataset.test.src=experiments/resources/data/my-dataset/test \
  --prediction_weights=checkpoint \
  --run-dir experiments/runs/my-student-predictions
```

`--prediction_weights` selects the model parameters:

- `checkpoint`: use EMA parameters when present, otherwise use raw parameters.
- `raw`: use the raw saved parameters.
- `ema`: require EMA parameters.

For ASE-based applications, [`student_calculator`](experiments/scripts/calculators.py) loads FAIR-Chem student checkpoints with an explicit raw/EMA choice. It maps GemNet and PaiNN scaling files to `experiments/runs/configs/scales/`. Update this mapping for other models or scaling files.

## Example configurations

The repository includes configurations from the paper for the following systems:

| Dataset | Subsets                                   | Teachers                                                             |
| ------- | ----------------------------------------- | -------------------------------------------------------------------- |
| SPICE   | `monomers`, `solvated`, `iodine`    | MACE-OFF                                                             |
| MPtrj   | `perovskites`, `yttrium`, `bandgap` | MACE-MP for all three subsets. eSEN-OAM for perovskites and yttrium. |

Student configurations cover GemNet-T, GemNet-dT, and PaiNN where reported. They are organized as:

```text
experiments/runs/accuracy/configs/<dataset>/<teacher>/<subset>/<objective>/<student>.yml
```

Shared dataset and model settings are in `experiments/runs/configs/`. The fixed example datasets and precomputed Hessian labels are available from the [Hessian distillation repository](https://github.com/ASK-Berkeley/MLFF-distill#downloading-the-data). Generate any additional labels required by your selected objective.

For example, after preparing the SPICE Solvated Amino Acids data and MACE-OFF labels:

```bash
python -m src.main --mode train \
  --config-yml experiments/runs/accuracy/configs/SPICE/MACE-OFF/solvated/joint/painn.yml \
  --run-dir experiments/runs/solvated-painn-joint \
  --seed 0
```

### Teacher evaluation and molecular dynamics

[Teacher evaluation](experiments/scripts/accuracy/teacher.py) computes energy and force metrics for SPICE and MPtrj.

```bash
python -m experiments.scripts.accuracy.teacher --help
```

The [MD script](experiments/scripts/md/simulate.py) runs ASE Langevin dynamics with periodic boundary conditions (`pbc=True`). Set the data and checkpoint paths in [its configuration](experiments/runs/md/config.yml) before running it.

```bash
python -m experiments.scripts.md.simulate \
  --config experiments/runs/md/config.yml \
  --model joint --index 0 \
  --output experiments/runs/md-joint-structure0
```

## Acknowledgments

This implementation builds on [Hessian distillation](https://github.com/ASK-Berkeley/MLFF-distill), [FAIR-Chem](https://github.com/FAIR-Chem/fairchem), and [MACE](https://github.com/ACEsuit/mace).

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{wang2027architecture,
  title   = {Architecture-agnostic knowledge distillation for machine-learning interatomic potentials},
  author  = {Wang, Chen and Hu, Siyu and Tan, Guangming and Jia, Weile},
  journal = {Neural Networks},
  volume  = {206},
  pages   = {109742},
  year    = {2027},
  doi     = {10.1016/j.neunet.2026.109742},
  url     = {https://doi.org/10.1016/j.neunet.2026.109742}
}
```
