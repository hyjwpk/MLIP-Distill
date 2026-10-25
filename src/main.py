"""
Modified from fairchem: https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/_cli.py

Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

from submitit import AutoExecutor
from submitit.helpers import Checkpointable, DelayedSubmission

from fairchem.core.common.flags import flags
from fairchem.core.common.utils import (
    build_config,
    create_grid,
    new_trainer_context,
    save_experiment_log,
    setup_logging,
)


if TYPE_CHECKING:
    import argparse


class Runner(Checkpointable):
    def __init__(self) -> None:
        self.config = None

    def __call__(self, config: dict) -> None:
        if config.get("mode") not in ("train", "predict"):
            raise ValueError("src.main only supports train and predict modes.")
        with new_trainer_context(config=config) as ctx:
            self.config = ctx.config
            self.task = ctx.task
            self.trainer = ctx.trainer
            self.task.setup(self.trainer)
            if self.config["mode"] == "predict":
                weights = self.config.get("prediction_weights", "checkpoint")
                if weights not in ("checkpoint", "raw", "ema"):
                    raise ValueError("prediction_weights must be checkpoint, raw or ema")
                if weights == "raw":
                    self.trainer.ema = None
                elif weights == "ema" and self.trainer.ema is None:
                    raise ValueError("Requested EMA parameters are absent from this checkpoint")
                logging.info("Prediction parameters: %s", "ema" if self.trainer.ema is not None else "raw")
            self.task.run()

    def checkpoint(self, *args, **kwargs):
        new_runner = Runner()
        self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.config["checkpoint"] = self.task.chkpt_path
        self.config["timestamp_id"] = self.trainer.timestamp_id
        if self.trainer.logger is not None:
            self.trainer.logger.mark_preempting()
        logging.info(
            f'Checkpointing callback is triggered, checkpoint saved to: {self.config["checkpoint"]}, timestamp_id: {self.config["timestamp_id"]}'
        )
        return DelayedSubmission(new_runner, self.config)


def main():
    """Train a model or predict with a saved checkpoint."""
    setup_logging()

    parser: argparse.ArgumentParser = flags.get_parser()
    args: argparse.Namespace
    override_args: list[str]
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    if config["mode"] not in ("train", "predict"):
        parser.error("src.main only supports --mode train or --mode predict.")

    if args.timestamp_id is not None and len(args.identifier) == 0:
        args.identifier = args.timestamp_id

    if args.submit:
        slurm_add_params = config.get("slurm", None)
        configs = create_grid(config, args.sweep_yml) if args.sweep_yml else [config]

        logging.info(f"Submitting {len(configs)} jobs")
        executor = AutoExecutor(folder=args.logdir / "%j", slurm_max_num_timeout=3)
        executor.update_parameters(
            name=args.identifier,
            mem_gb=args.slurm_mem,
            timeout_min=args.slurm_timeout * 60,
            slurm_partition=args.slurm_partition,
            gpus_per_node=args.num_gpus,
            cpus_per_task=(config["optim"]["num_workers"] + 1),
            tasks_per_node=(args.num_gpus),
            nodes=args.num_nodes,
            slurm_additional_parameters=slurm_add_params,
        )
        for config in configs:
            config["slurm"] = copy.deepcopy(executor.parameters)
            config["slurm"]["folder"] = str(executor.folder)
        jobs = executor.map_array(Runner(), configs)
        logging.info(f"Submitted jobs: {', '.join([job.job_id for job in jobs])}")
        log_file = save_experiment_log(args, jobs, configs)
        logging.info(f"Experiment log saved to: {log_file}")

    else:
        Runner()(config)


if __name__ == "__main__":
    main()
