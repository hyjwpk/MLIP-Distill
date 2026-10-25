"""
Generate eSEN teacher labels through FAIR-Chem.

Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import os
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import lmdb

from fairchem.core.common import distutils
from fairchem.core.common.registry import registry
from fairchem.core.modules.exponential_moving_average import ExponentialMovingAverage
from fairchem.core.trainers.ocp_trainer import OCPTrainer
from src.distill_utils import get_teacher_jacobian


@contextlib.contextmanager
def force_derivative_graph(model):
    """Retain force derivatives without changing model training/evaluation mode."""
    heads = [module for module in model.modules() if hasattr(module, "force_create_graph")]
    if not heads:
        raise RuntimeError("The eSEN model requires FAIR-Chem with force_create_graph support")
    previous = [head.force_create_graph for head in heads]
    try:
        for head in heads:
            head.force_create_graph = True
        with torch.enable_grad():
            yield
    finally:
        for head, value in zip(heads, previous):
            head.force_create_graph = value


@registry.register_trainer("labels")
class LabelsTrainer(OCPTrainer):
    def run(self):
        """Generate teacher labels after trainer initialization."""
        labels_folder = self.config['dataset']['teacher_labels_folder']
        model_attributes_holder = self.config['model']

        checkpoint_path = self.config["dataset"]["teacher_checkpoint_path"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.teacher_config = checkpoint["config"]
        model_config = copy.deepcopy(self.teacher_config["model"])
        if isinstance(model_config, str):
            model_config = copy.deepcopy(self.teacher_config["model_attributes"])
            model_config["name"] = self.teacher_config["model"]

        scale_file = self.config["dataset"].get("teacher_scale_file")
        if scale_file:
            model_config["scale_file"] = scale_file
        else:
            # Scaling values are restored from the checkpoint, not its old file path.
            model_config.pop("scale_file", None)

        weights = self.config["dataset"].get("teacher_weights", "checkpoint")
        if weights not in ("checkpoint", "raw", "ema"):
            raise ValueError("teacher_weights must be checkpoint, raw or ema")
        ema_state = checkpoint.get("ema")
        if weights == "ema" and ema_state is None:
            raise ValueError("Requested EMA parameters are absent from the teacher checkpoint")
        use_ema = weights != "raw" and ema_state is not None
        if not use_ema:
            checkpoint = {**checkpoint, "ema": None}

        self.prepare_label_directories(labels_folder)

        try:
            self.config['model'] = model_config
            self.normalizers = {}  # Restored from the teacher checkpoint below.
            self.load_task()
            self.load_model()
            # EMA must reference the teacher parameters, not the initial model.
            self.ema = (ExponentialMovingAverage(self.model.parameters(), ema_state["decay"])
                        if use_ema else None)
            self.load_checkpoint(checkpoint_path, checkpoint=checkpoint, inference_only=True)
            if use_ema:
                self.ema.copy_to()
            self.model.eval()
            logging.info("Teacher label parameters: %s", "ema" if use_ema else "raw")

            self.launch_record_tasks(labels_folder, 'train')
            self.launch_record_tasks(labels_folder, 'val')
        finally:
            self.config['model'] = model_attributes_holder

    def prepare_label_directories(self, labels_folder):
        """Create shared output directories once and report failures to every rank."""
        error = [None]
        if distutils.is_master():
            try:
                if os.path.islink(labels_folder) or (
                    os.path.lexists(labels_folder) and not os.path.isdir(labels_folder)
                ):
                    raise ValueError(f"Output must be a directory, not a file or symlink: {labels_folder}")
                folders = [os.path.join(labels_folder, name) for name in (
                    "train_forces", "val_forces", "force_jacobians", "node_energy"
                )]
                for folder in folders:
                    if os.path.lexists(folder):
                        raise FileExistsError(f"Refusing to overwrite labels: {folder}")
                os.makedirs(labels_folder, exist_ok=True)
                for folder in folders:
                    os.makedirs(folder)
            except (OSError, ValueError) as exc:
                error[0] = f"{type(exc).__name__}: {exc}"
        # Broadcast only after creation, so no worker opens a shard too early.
        distutils.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError(f"Cannot prepare label output: {error[0]}")

    def launch_record_tasks(self, labels_folder, dataset_type):
        """Assign each existing worker a contiguous range of the split."""
        rank = distutils.get_rank()
        world_size = distutils.get_world_size()

        dataset_size = len(self.train_dataset if dataset_type == 'train' else self.val_dataset)
        indices_per_worker = dataset_size // world_size
        start_idx = rank * indices_per_worker
        end_idx = start_idx + indices_per_worker if rank < world_size - 1 else dataset_size

        self.record_labels_parallel(labels_folder, dataset_type, start_idx, end_idx)
    
    def record_labels_parallel(self, labels_folder, dataset_type, start_idx, end_idx):
        """Write this worker's labels to separate LMDB shards."""
        if dataset_type == 'train':
            dataset = self.train_dataset
        elif dataset_type == 'val':
            dataset = self.val_dataset
        else:
            raise ValueError("Invalid dataset type provided")

        subset_indices = list(range(start_idx, end_idx))
        subset_dataset = Subset(dataset, subset_indices)
        dataloader = DataLoader(
            subset_dataset,
            collate_fn=self.collater,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
            batch_size=self.config["dataset"]["label_force_batch_size"],
        )
        
        lmdb_path = os.path.join(labels_folder, f"{dataset_type}_forces", f"data.{distutils.get_rank():04d}.lmdb")

        def get_separated_forces(batch):
            all_forces = self._forward(batch)['forces']
            all_forces = self._denorm_preds('forces', all_forces, batch)
            natoms = batch.natoms
            return [all_forces[sum(natoms[:i]):sum(natoms[:i+1])] for i in range(len(natoms))]
        
        self.record_and_save(dataloader, lmdb_path, get_separated_forces, start_idx)

        if dataset_type == 'train':
            lmdb_path = os.path.join(labels_folder, "node_energy", f"data.{distutils.get_rank():04d}.lmdb")

            def get_node_energy(batch):
                all_node_energy = self._forward(batch)['node_energy']
                all_node_energy = self.normalizers['energy'](all_node_energy)
                all_node_energy = self.elementrefs["energy"].apply_atomwise_refs(all_node_energy, batch, 1)
                natoms = batch.natoms
                return [all_node_energy[sum(natoms[:i]):sum(natoms[:i+1])] for i in range(len(natoms))]

            self.record_and_save(dataloader, lmdb_path, get_node_energy, start_idx)

        if dataset_type == 'train':
            jac_dataloader = DataLoader(
            subset_dataset,
            collate_fn=self.collater,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
            batch_size=self.config["dataset"]["label_jac_batch_size"],
            )
            
            jac_lmdb_path = os.path.join(labels_folder, "force_jacobians", f"data.{distutils.get_rank():04d}.lmdb")
            should_mask = self.output_targets['forces']["train_on_free_atoms"]
            def get_separated_force_jacs(batch): 
                with force_derivative_graph(self.model):
                    forces = self._denorm_preds('forces', self._forward(batch)['forces'], batch)
                    return get_teacher_jacobian(forces, batch, vectorize=self.config["dataset"]["vectorize_teach_jacs"], should_mask=should_mask)
            self.record_and_save(jac_dataloader, jac_lmdb_path, get_separated_force_jacs, start_idx)

    def record_and_save(self, dataloader, file_path, fn, start_idx):
        map_size = 1099511627776 * 2
        if os.path.lexists(file_path):
            raise FileExistsError(f"Refusing to overwrite label shard: {file_path}")
        env = lmdb.open(file_path, map_size=map_size)
        try:
            no_sync_context = self.model.no_sync() if distutils.get_world_size() > 1 else contextlib.nullcontext()
            next_idx = start_idx
            with no_sync_context:
                for batch in tqdm(dataloader):
                    batch_output = fn(batch)
                    if len(batch_output) != len(batch.natoms):
                        raise ValueError("Expected one label array per structure")
                    with env.begin(write=True) as txn:
                        for output in batch_output:
                            # Keys match the fixed split order, not source-file sample IDs.
                            txn.put(str(next_idx).encode(), output.detach().cpu().numpy().tobytes())
                            next_idx += 1
        finally:
            env.close()
        logging.info(f"All tensors saved to LMDB:{file_path}")


def main():
    """Generate eSEN labels using the fixed dataset splits in a YAML config."""
    from fairchem.core.common.flags import flags
    from fairchem.core.common.utils import build_config, new_trainer_context, setup_logging

    setup_logging()
    parser = flags.get_parser()
    parser.description = __doc__
    parser.set_defaults(mode="train")
    args, overrides = parser.parse_known_args()
    if args.submit or args.sweep_yml:
        parser.error("Launch this script directly or with torchrun; submission and sweeps are not supported.")
    config = build_config(args, overrides)
    if config["mode"] != "train":
        parser.error("This entry point only generates labels; omit --mode.")
    if config.get("checkpoint"):
        parser.error("Set dataset.train.teacher_checkpoint_path instead of --checkpoint.")
    train = config["dataset"].get("train", config["dataset"])
    required = ("teacher_checkpoint_path", "teacher_labels_folder",
                "label_force_batch_size", "label_jac_batch_size", "vectorize_teach_jacs")
    missing = [key for key in required if key not in train]
    if missing:
        parser.error("Missing teacher-label settings: " + ", ".join(missing))
    config["trainer"] = "labels"
    with new_trainer_context(config=config) as ctx:
        ctx.trainer.run()


if __name__ == "__main__":
    main()
