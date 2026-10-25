"""Evaluate a local teacher checkpoint on one existing dataset split."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


class Metrics:
    def __init__(self):
        self.structures = 0
        self.atoms = 0
        self.energy_error = 0.0
        self.per_atom_energy_error = 0.0
        self.force_error = 0.0

    def update(self, energy, reference_energy, forces, reference_forces):
        forces = np.asarray(forces, dtype=np.float64)
        reference_forces = np.asarray(reference_forces, dtype=np.float64)
        if forces.ndim != 2 or forces.shape[1] != 3 or not len(forces):
            raise ValueError("Forces must have shape (N, 3), with N > 0")
        if forces.shape != reference_forces.shape:
            raise ValueError("Predicted and reference force shapes differ")
        if not (np.isfinite(energy) and np.isfinite(reference_energy)
                and np.isfinite(forces).all() and np.isfinite(reference_forces).all()):
            raise ValueError("Non-finite energy or force value")
        error = abs(float(energy) - float(reference_energy))
        self.structures += 1
        self.atoms += len(forces)
        self.energy_error += error
        self.per_atom_energy_error += error / len(forces)
        self.force_error += np.abs(forces - reference_forces).mean()

    def result(self, family):
        if family not in ("spice", "mptrj"):
            raise ValueError(f"Unknown dataset family: {family}")
        if not self.structures:
            raise ValueError("Cannot evaluate an empty dataset")
        energy = self.per_atom_energy_error if family == "spice" else self.energy_error
        return {
            "structures": self.structures,
            "atoms": self.atoms,
            "energy_mae": energy / self.structures * 1000,
            "energy_unit": "meV/atom" if family == "spice" else "meV",
            "force_mae": float(self.force_error / self.structures * 1000),
            "force_unit": "meV/angstrom",
            "energy_aggregation": "mean_over_structures",
            "force_aggregation": "mean_over_structures_of_cartesian_component_mae",
        }


def evaluate(dataset, calculator, family):
    from ase import Atoms

    metrics = Metrics()
    for sample in dataset:
        periodic = family == "mptrj"
        atoms = Atoms(
            numbers=np.asarray(sample.atomic_numbers, dtype=int),
            positions=np.asarray(sample.pos),
            cell=np.asarray(sample.cell).reshape(3, 3) if periodic else None,
            pbc=periodic,
        )
        reference_energy = sample.uncorrected_total_energy if periodic else sample.y
        atoms.calc = calculator
        metrics.update(
            atoms.get_potential_energy(), reference_energy.item(),
            atoms.get_forces(), np.asarray(sample.force),
        )
    return metrics.result(family)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True, choices=("mace-off", "mace-mp", "esen"))
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Existing subset directory containing train/val/test")
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    dataset_path = args.dataset_root / args.split
    if not dataset_path.is_dir():
        parser.error(f"Dataset split does not exist: {dataset_path}")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if os.path.lexists(args.output):
        parser.error(f"Refusing to overwrite output: {args.output}")

    from fairchem.core.datasets import LmdbDataset

    if args.teacher == "esen":
        from fairchem.core import OCPCalculator
        calculator = OCPCalculator(checkpoint_path=str(args.checkpoint), cpu=args.device == "cpu")
    else:
        from mace.calculators import mace_mp, mace_off
        factory = mace_off if args.teacher == "mace-off" else mace_mp
        calculator = factory(model=str(args.checkpoint), dispersion=False,
                             default_dtype="float32", device=args.device)
    family = "spice" if args.teacher == "mace-off" else "mptrj"
    dataset = LmdbDataset({"src": str(dataset_path)})
    try:
        result = evaluate(dataset, calculator, family)
    finally:
        del dataset
    result.update({
        "teacher": args.teacher,
        "checkpoint": str(args.checkpoint),
        "dataset": str(dataset_path),
        "split": args.split,
        "device": args.device,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
