from torch.utils.data import Dataset
import os
from neural_paw_dft.spin_electrafi.tools.density_conversions import get_density
import numpy as np
from torch.utils.data import IterableDataset
import random
import torch

def collate_fn(batch): return batch

class DensityDataset(Dataset):
    def __init__(self, file_list, dens_path, config, ood_name=None):
        self.file_list = file_list
        self.dens_path = dens_path
        self.config = config
        self.cache = {}
        # Global cap for the cache (number of items). 0 or None => no caching.
        self._max_cache_items = 40000
        self._announced_full = False  # just for one-time print when full
        self.ood_name = ood_name

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        valid = False
        while not valid:
            try:
                cd_file = self.file_list[idx]
                path = os.path.join(self.dens_path, cd_file)

                # Serve from cache if present
                if cd_file in self.cache:
                    return self.cache[cd_file]

                density, struc, n_elec, grid_dict, cd_diff = get_density(path)
                valid = True
            except Exception as e:
                print(f"Error loading {cd_file}: {e}")
                idx = np.random.randint(0, max(1, len(self.file_list) - 1))
        if self.ood_name is None:
            sample = (density, struc, n_elec, grid_dict, cd_file, None, cd_diff)
        else:
            sample = (density, struc, n_elec, grid_dict, cd_file, self.ood_name, cd_diff)

        # Cache only if:
        #  - not in save_memory mode
        #  - global cap > 0
        #  - cache not yet full
        if (not self.config.get("save_memory", False)
            and self._max_cache_items > 0
            and len(self.cache) < self._max_cache_items):
            self.cache[cd_file] = sample
            if len(self.cache) == self._max_cache_items and not self._announced_full:
                self._announced_full = True
                print(f"[DensityDataset] Cache filled to cap: {self._max_cache_items} items.")
            return self.cache[cd_file]

        # Otherwise return without caching
        return sample

def _dl_worker_init(_):
    # keep temp CHGCAR files on tmpfs (RAM)
    os.environ.setdefault("TMPDIR", "/dev/shm")
    # free CPU for I/O/decompress; PyTorch ops shouldn’t hog threads in workers
    import torch
    torch.set_num_threads(1)

# OBS: ADDING A LEN PROPERTY CAUSES ISSUES WITH MULTI-WORKER DATALOADERS IN LIGHTNING FOR ITERABLEDATASET
class DensityStream(IterableDataset):
    def __init__(self, file_list, dens_path, config, ood_name=None, shuffle_each_epoch=False):
        self.file_list = list(file_list)
        self.dens_path = dens_path
        self.config = config
        self.ood_name = ood_name
        self.shuffle_each_epoch = shuffle_each_epoch

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        files = self.file_list
        if self.shuffle_each_epoch:
            rng = random.Random(torch.randint(0, 10**9, ()).item())
            rng.shuffle(files)

        if worker is None:
            # single-process
            yield from self._iter_files(files)
        else:
            # shard: each worker gets a slice
            per_worker = int((len(files) + worker.num_workers - 1) / worker.num_workers)
            start = worker.id * per_worker
            end   = min(start + per_worker, len(files))
            yield from self._iter_files(files[start:end])

    def _iter_files(self, files):
        for cd_file in files:
            path = os.path.join(self.dens_path, cd_file)
            try:
                density, struc, n_elec, grid_dict, cd_diff = get_density(path)  # your lz4->tmp->read
                if self.ood_name is None:
                    yield (density, struc, n_elec, grid_dict, cd_file, None, cd_diff)
                else:
                    yield (density, struc, n_elec, grid_dict, cd_file, self.ood_name, cd_diff)
            except Exception as e:
                print(f"[skip] {cd_file}: {e}")
                continue