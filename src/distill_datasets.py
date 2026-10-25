import bisect
import os
from torch.utils.data import Dataset, Subset
import lmdb
import torch
import numpy as np

class CombinedDataset(Dataset):
    def __init__(self, main_dataset, teach_force_dataset=None, node_energy_dataset=None, force_jac_dataset=None):
        for name, dataset in (("Teacher forces", teach_force_dataset),
                              ("Atomic energies", node_energy_dataset),
                              ("Force Jacobians", force_jac_dataset)):
            if dataset is not None and len(main_dataset) != len(dataset):
                raise ValueError(f"{name}: {len(dataset)} labels for {len(main_dataset)} structures")
        self.main_dataset = main_dataset
        self.teach_force_dataset = teach_force_dataset
        self.node_energy_dataset = node_energy_dataset
        self.force_jac_dataset = force_jac_dataset 
        self._metadata = self.main_dataset._metadata
        self.metadata_hasattr = self.main_dataset.metadata_hasattr
        
        self.get_metadata = self.main_dataset.get_metadata

    def __len__(self):
        return len(self.main_dataset)

    def __getitem__(self, idx):
        main_batch = self.main_dataset[idx]
        num_atoms = main_batch.natoms
        teacher_forces = (self.teach_force_dataset[idx].reshape(num_atoms, 3)
                          if self.teach_force_dataset is not None else None)
        if self.node_energy_dataset is not None:
            node_energy = self.node_energy_dataset[idx].reshape(num_atoms, 1)
        else:
            node_energy = None
        if self.force_jac_dataset is not None:
            # Keep variable-size Jacobians flat until the batched loss reshapes them.
            force_jacs = self.force_jac_dataset[idx]
        else: 
            force_jacs = None
            
        
        if not hasattr(main_batch, 'cell'):
            main_batch.cell =  torch.eye(3, device=main_batch.pos.device).unsqueeze(0)
        if not hasattr(main_batch, 'fixed'):
            main_batch.fixed = torch.zeros(main_batch.pos.shape[0], device=main_batch.pos.device) 
        main_batch.teacher_forces = teacher_forces
        main_batch.node_energy = node_energy
        main_batch.force_jacs = force_jacs
        return main_batch

    def close_db(self):
        for dataset in (self.main_dataset, self.teach_force_dataset,
                        self.force_jac_dataset, self.node_energy_dataset):
            if dataset is None:
                continue
            while isinstance(dataset, Subset):
                dataset = dataset.dataset
            close_db = getattr(dataset, "close_db", None)
            if callable(close_db):
                close_db()
            else:
                # FAIR-Chem LmdbDataset exposes envs (directory) or env (single file).
                for env in getattr(dataset, "envs", []):
                    env.close()
                env = getattr(dataset, "env", None)
                if env is not None:
                    env.close()

class SimpleDataset(Dataset):
    def __init__(self, folder_path, dtype=np.float32):
        self.folder_path = folder_path
        self.dtype = dtype
        
        self.db_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.lmdb')])
        assert len(self.db_paths) > 0, "No LMDB files found in the specified folder."

        self.envs = []
        self._keys = []
        self._keylen_cumulative = []

        total_entries = 0
        for db_path in self.db_paths:
            env = lmdb.open(db_path, readonly=True, lock=False)
            self.envs.append(env)
            with env.begin() as txn:
                num_entries = txn.stat()['entries']
                total_entries += num_entries
                self._keylen_cumulative.append(total_entries)

    def __len__(self):
        return self._keylen_cumulative[-1] if self._keylen_cumulative else 0

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()

        db_idx = bisect.bisect_right(self._keylen_cumulative, index)

        with self.envs[db_idx].begin() as txn:
            byte_data = txn.get(str(index).encode())
            if byte_data:
                tensor = torch.from_numpy(np.frombuffer(byte_data, dtype=self.dtype))
                return tensor
            else:
                raise Exception(f"Data not found for index {index} in LMDB file.")


    def close_db(self):
        for env in self.envs:
            env.close()
