"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import time
import numpy as np

import torch
from torch.utils.data import Subset
from tqdm import tqdm

from fairchem.core.common import distutils 
from fairchem.core.common.registry import registry


from fairchem.core.trainers.ocp_trainer import OCPTrainer
from . import get_hessian_loss, print_cuda_memory_usage
from . import CombinedDataset, SimpleDataset


@registry.register_trainer("distill")
class DistillTrainer(OCPTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_validating = False
        self.force_mae =  None
        self.node_energy_loss_fn = self.loss_functions.pop()
        self.hessian_loss_fn = self.loss_functions.pop()
        if self.loss_functions[-1][0] == 'teacher_forces':
            teacher_force_loss_fn = self.loss_functions.pop()
            if teacher_force_loss_fn[1]['coefficient'] != 0:
                raise ValueError('Direct teacher-force distillation is no longer supported.')
        assert self.node_energy_loss_fn[0] == 'node_energy'
        assert self.hessian_loss_fn[0] == 'hessian'
        self.original_hessian_coeff = self.hessian_loss_fn[1]['coefficient']
        self.modify_train_val_datasets()
        self.start_time = time.time()
        if self.original_hessian_coeff > 0:
            self.calculate_teacher_loss()
    
    def calculate_teacher_loss(self):
        self.teacher_force_mae = 0
        for datapoint in tqdm(self.val_dataset):
            true_label = datapoint['forces'].to(self.device)
            # Validation metrics and stored teacher forces use physical units.
            self.teacher_force_mae += torch.abs(datapoint['teacher_forces'].to(self.device) - true_label).mean().item()
        self.teacher_force_mae /= len(self.val_dataset)
        logging.info("Teacher force MAE: %s", self.teacher_force_mae)
        
    def modify_train_val_datasets(self):
        train_indxs = val_indxs = None
        if "split" in self.config["dataset"]:
            train_indxs = np.random.default_rng(seed=0).choice(
                    len(self.train_dataset), 
                    self.config["dataset"]["split"], 
                    replace=False
            )
        if "split" in self.config["val_dataset"]:
            val_indxs = np.random.default_rng(seed=0).choice(
                len(self.val_dataset), 
                self.config["val_dataset"]["split"], 
                replace=False
            )
        self.train_dataset = self.insert_teach_datasets(self.train_dataset, 'train', train_indxs )

        self.train_sampler = self.get_sampler(
            self.train_dataset,
            self.config["optim"]["batch_size"],
            shuffle=True,
        )
        self.train_loader = self.get_dataloader(
            self.train_dataset,
            self.train_sampler,
        )

        self.val_dataset = self.insert_teach_datasets(self.val_dataset, 'val', val_indxs)

        self.val_sampler = self.get_sampler(
            self.val_dataset,
            self.config["optim"].get(
                "eval_batch_size", self.config["optim"]["batch_size"]
            ),
            shuffle=False,
        )
        self.val_loader = self.get_dataloader(
            self.val_dataset,
            self.val_sampler,
        )

    def insert_teach_datasets(self, main_dataset, dataset_type, indxs=None):
        labels_folder = self.config['dataset']['teacher_labels_folder']

        def load_labels(name):
            dataset = SimpleDataset(os.path.join(labels_folder, name))
            if indxs is not None:
                dataset = Subset(dataset, torch.tensor(indxs))
            return dataset

        teacher_force_dataset = node_energy_dataset = force_jac_dataset = None
        if dataset_type == 'train':
            if self.node_energy_loss_fn[1]['coefficient'] != 0:
                node_energy_dataset = load_labels('node_energy')
            if self.original_hessian_coeff > 0:
                force_jac_dataset = load_labels('force_jacobians')
        elif dataset_type == 'val':
            # Teacher validation forces set the threshold for the Hessian weight.
            if self.original_hessian_coeff > 0:
                teacher_force_dataset = load_labels('val_forces')
        else:
            raise ValueError(f"Unsupported dataset split: {dataset_type}")
        return CombinedDataset(main_dataset,  teacher_force_dataset, node_energy_dataset, force_jac_dataset)

    def update_loss_coefficients(self):
        if self.original_hessian_coeff <= 0:
            return
        if self.force_mae == None:
            self.force_mae = float('inf')
        if self.step % 20 == 0:
            if self.force_mae < self.teacher_force_mae:
                logging.info("Student force MAE is lower than teacher force MAE.")

            percent_higher = ((self.force_mae - self.teacher_force_mae) / self.teacher_force_mae) *100
            threshold = 5
            if percent_higher < threshold: 
                self.hessian_loss_fn[1]['coefficient'] = 0.5 * self.original_hessian_coeff
           
    def _forward(self, batch):
        if not self.is_validating:
            batch.pos.requires_grad_(True)
        return super()._forward(batch)

    def _normalize_atomic_energies(self, target, batch):
        """Use the same reference and scale as the total-energy target."""
        target = target.reshape(-1, 1)
        if "energy" in self.elementrefs:
            target = self.elementrefs["energy"].apply_atomwise_refs(target, batch, -1)
        if "energy" in self.normalizers:
            normalizer = self.normalizers["energy"]
            natoms = batch.natoms[batch.batch].reshape(-1, 1)
            # Split the system-level offset across atoms so their sum is preserved.
            target = (target - normalizer.mean / natoms) / normalizer.rmsd
        return target

    def validate(self, split: str = "val", disable_tqdm: bool = False):
        self.is_validating =True 

        total_rate =  self.step / (time.time() - self.start_time)  * 60
        logging.info("Average training rate: %.2f iterations/min", total_rate)

        val_metrics = super().validate(split, disable_tqdm)

        self.force_mae = val_metrics['forces_mae']['metric']

        self.is_validating = False
        return val_metrics

    def _compute_loss(self, out, batch):
        if self.is_validating:
            return torch.tensor([0])
        fixed = batch.fixed
        mask = fixed == 0
        should_mask = self.output_targets['forces']["train_on_free_atoms"]
        self.update_loss_coefficients()
        
        loss = [super()._compute_loss(out, batch)]
        batch['hessian_loss'] = torch.tensor(0)

        if self.hessian_loss_fn[1]['coefficient'] > 0:
            hessian_loss = get_hessian_loss(
                out=out, 
                batch=batch, 
                num_samples=self.config['optim']['hessian_sample_size'], 
                mask= mask, 
                should_mask=should_mask, 
                finite_differences= self.config['optim'].get('finite_differences', False),
                looped=(not self.config['optim']["vectorize_jacs"]),
                collater = self.collater,
                forward = self._forward,
                force_scale=(self.normalizers['forces'].rmsd
                             if 'forces' in self.normalizers else 1.0),
            )
            if self.config['optim'].get("print_memory_usage", False):
                print_cuda_memory_usage()
            loss.append(hessian_loss * self.hessian_loss_fn[1]['coefficient'])
            batch['hessian_loss'] = hessian_loss

        if self.node_energy_loss_fn[1]['coefficient'] != 0:
            natoms = torch.repeat_interleave(batch.natoms, batch.natoms)
            
            mult = self.node_energy_loss_fn[1]["coefficient"]

            node_energy_out = out['node_energy']
            node_energy_batch = self._normalize_atomic_energies(batch['node_energy'], batch)

            curr_loss = self.node_energy_loss_fn[1]["fn"](
                        node_energy_out,
                        node_energy_batch,
                        natoms=natoms,
            )
            batch['node_energy_loss'] = curr_loss
            loss.append(
                mult
                * curr_loss
            )
        for lc in loss:
            if torch.any(torch.isnan(lc)):
                raise ValueError("Training loss contains NaN values")
            assert hasattr(lc, "grad_fn")
        return sum(loss)
    
    def _compute_metrics(self, out, batch, evaluator, metrics=None):
        metrics = super()._compute_metrics(out, batch, evaluator, metrics)
        if not self.is_validating and self.node_energy_loss_fn[1]['coefficient'] > 0:
            avg_node_energy_loss = distutils.all_reduce(batch["node_energy_loss"], average=True )
            metrics['node_energy_loss'] = {}
            metrics['node_energy_loss']['metric'] = avg_node_energy_loss.item()
            metrics['node_energy_loss']['total'] = avg_node_energy_loss.item()
        if not self.is_validating and self.original_hessian_coeff > 0:
            avg_hessian_loss = distutils.all_reduce(batch["hessian_loss"], average=True )
            metrics['hessian_loss'] = {}
            metrics['hessian_loss']['metric'] = avg_hessian_loss.item()
            metrics['hessian_loss']['total'] = avg_hessian_loss.item()
        return metrics
