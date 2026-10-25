"""Run one archived MD model on a fixed SPICE test structure."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.logger import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from fairchem.core.datasets import LmdbDataset

from experiments.scripts.calculators import student_calculator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, help="Override the production length for a short test")
    parser.add_argument("--inference-only", action="store_true", help="Evaluate the initial structure without MD")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.model not in config["models"]:
        parser.error(f"Unknown model. Choose from {', '.join(config['models'])}")
    steps = config["steps"] if args.steps is None else args.steps
    if steps < 0 or args.index < 0:
        parser.error("steps and index must be nonnegative")
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"Refusing to reuse output: {args.output}")
    data = LmdbDataset({"src": config["dataset"]})
    try:
        sample = data[args.index]
        atoms = Atoms(numbers=sample.atomic_numbers.numpy(), positions=sample.pos.numpy(),
                      cell=sample.cell.numpy().reshape(3, 3), pbc=True)
    finally:
        for env in data.envs:
            env.close()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = config["models"][args.model]
    if model["type"] == "mace":
        from mace.calculators import MACECalculator
        atoms.calc = MACECalculator(model_paths=model["checkpoint"], device="cuda", default_dtype="float32")
    else:
        atoms.calc = student_calculator(model["checkpoint"], model["weights"], args.seed)
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
    if not (np.isfinite(energy) and np.isfinite(forces).all()):
        raise ValueError("Non-finite initial prediction")
    print(f"model={args.model} checkpoint={model['checkpoint']} weights={model.get('weights', 'native')}")
    print(f"initial_energy_eV={energy} max_abs_force_eV_A={np.abs(forces).max()}", flush=True)
    if args.inference_only:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.RandomState(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=config["temperature"], rng=rng)
    Stationary(atoms)
    dynamics = Langevin(atoms, timestep=config["timestep_fs"] * units.fs,
                        temperature_K=config["temperature"], friction=config["friction"] / units.fs,
                        rng=rng, trajectory=str(args.output / f"equilibration{args.index}.traj"))
    dynamics.attach(MDLogger(dynamics, atoms, str(args.output / f"md_system{args.index}_seed{args.seed}.log"),
                            header=True, stress=False, peratom=True, mode="w"), interval=config["save_freq"])
    dynamics.run(steps)
    print(f"Completed {steps} MD steps", flush=True)


if __name__ == "__main__":
    main()
