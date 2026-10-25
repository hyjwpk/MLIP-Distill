"""Generate selected MACE-OFF labels for the fixed train/validation splits."""

import argparse
import os
from pathlib import Path

import lmdb
import numpy as np
from ase import Atoms
from tqdm import tqdm


LABELS = {
    "node-energy": ("train", "node_energy"),
    "hessian": ("train", "force_jacobians"),
    "train-forces": ("train", "train_forces"),
    "val-forces": ("val", "val_forces"),
}


def record_and_save(dataset, file_path, fn):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=False)
    env = lmdb.open(str(file_path), map_size=2 * 1024**4)
    try:
        with env.begin(write=True) as txn:
            for index, sample in enumerate(tqdm(dataset)):
                sample_id = str(index)
                values = np.asarray(fn(sample), dtype=np.float32)
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite label for sample {sample_id}")
                txn.put(sample_id.encode(), values.tobytes())
    finally:
        env.close()
    print(f"Labels saved to {file_path}", flush=True)


def record_labels(labels_folder, dataset_path, model, labels=("node-energy",)):
    from fairchem.core.datasets import LmdbDataset
    from mace.calculators import mace_off

    labels_folder = Path(labels_folder)
    dataset_path = Path(dataset_path)
    model = Path(model)
    selected = tuple(dict.fromkeys(labels))
    if not selected or any(label not in LABELS for label in selected):
        raise ValueError(f"Choose labels from: {', '.join(LABELS)}")
    if labels_folder.is_symlink() or (labels_folder.exists() and not labels_folder.is_dir()):
        raise ValueError(f"Output must be a directory, not a file or symlink: {labels_folder}")
    for label in selected:
        split, folder = LABELS[label]
        if not (dataset_path / split).is_dir():
            raise FileNotFoundError(dataset_path / split)
        if os.path.lexists(labels_folder / folder):
            raise FileExistsError(f"Refusing to overwrite labels: {labels_folder / folder}")
    if not model.is_file():
        raise FileNotFoundError(model)
    calc = mace_off(model=str(model), dispersion=False, default_dtype="float32", device="cuda")

    def sample_to_atoms(sample):
        return Atoms(numbers=sample.atomic_numbers.numpy(), positions=sample.pos.numpy())

    def get_forces(sample):
        atoms = sample_to_atoms(sample)
        atoms.calc = calc
        return atoms.get_forces()

    def get_hessians(sample):
        atoms = sample_to_atoms(sample)
        natoms = len(atoms)
        # Training stores the force Jacobian, which is the negative energy Hessian.
        return -np.asarray(calc.get_hessian(atoms=atoms)).reshape(natoms, 3, natoms, 3)

    def get_node_energy(sample):
        atoms = sample_to_atoms(sample)
        atoms.calc = calc
        atoms.get_potential_energy()
        return calc.results["node_energy"]

    generators = {"node-energy": get_node_energy, "hessian": get_hessians,
                  "train-forces": get_forces, "val-forces": get_forces}
    labels_folder.mkdir(parents=True, exist_ok=True)
    datasets = {}
    try:
        for label in selected:
            split, folder = LABELS[label]
            if split not in datasets:
                datasets[split] = LmdbDataset({"src": str(dataset_path / split)})
            print(f"Generating {label} from {split}", flush=True)
            record_and_save(datasets[split], labels_folder / folder / f"{folder}.lmdb",
                            generators[label])
    finally:
        for dataset in datasets.values():
            for env in dataset.envs:
                env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Local teacher checkpoint")
    parser.add_argument("--output", type=Path, required=True,
                        help="Label root; existing unselected label directories are preserved")
    parser.add_argument("--labels", nargs="+", choices=tuple(LABELS), default=["node-energy"],
                        help="Outputs to generate; Hessian and atomic energies use train only")
    args = parser.parse_args()
    record_labels(args.output, args.dataset, args.model, args.labels)
