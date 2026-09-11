from torch.optim import Adam, AdamW, SGD
import torch
import numpy as np
import random
import os
import csv
import json
from .muon import Muon
from neural_init._resources import resolve_data_path
import torch.nn as nn

def set_optimizer(tens_net, config):
    if config['optimizer'] == 'adam':
        optimizer = Adam(tens_net.parameters(),
                         lr=config['initial_lr'],
                         amsgrad=config["amsgrad"],
                         weight_decay=config['weight_decay'],
                         fused=config['fused'])
    elif config['optimizer'] == 'adamw':
        optimizer = AdamW(tens_net.parameters(),
                          lr=config['initial_lr'],
                          amsgrad=config["amsgrad"],
                          weight_decay=config['weight_decay'],
                          fused=config['fused'])
    elif config['optimizer'] == 'sgd':
        optimizer = SGD(tens_net.parameters(),
                        lr=config['initial_lr'],
                        momentum=config['momentum'],
                        weight_decay=config['weight_decay'])
    return optimizer

def set_optimizers_mixed(model, config):
    muon_params, adamw_params = split_params_for_muon(model)

    # You can also make weight-decay-free subgroups for biases/norms if you want
    optimizers = []
    if muon_params:
        opt_muon = Muon(
            muon_params,
            lr=config["initial_lr"]*10,  # typically use higher LR for Muon
            weight_decay=config['weight_decay']
        )
        optimizers.append(opt_muon)

    if adamw_params:
        opt_adamw = AdamW(adamw_params,
                          lr=config['initial_lr'],
                          amsgrad=config["amsgrad"],
                          weight_decay=config['weight_decay'],
                          fused=config['fused'])
        optimizers.append(opt_adamw)

    return optimizers

def split_params_for_muon(model: nn.Module):
    muon_params, adamw_params = [], []
    picked = set()

    # 1) Put ONLY Linear weights on Muon
    for mod_name, mod in model.named_modules():
        # Plain Linear or NonDynamicallyQuantizableLinear are fine
        if isinstance(mod, nn.Linear):
            # pick only the weight (2D)
            p = getattr(mod, "weight", None)
            if p is not None and p.requires_grad and p.ndim == 2:
                muon_params.append(p)
                picked.add(id(p))
            # bias stays on AdamW
            b = getattr(mod, "bias", None)
            if b is not None and b.requires_grad:
                adamw_params.append(b)
                picked.add(id(b))

        # Explicitly keep these on AdamW
        if isinstance(mod, (nn.Embedding,
                            nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for pname, p in mod.named_parameters(recurse=False):
                if p.requires_grad and id(p) not in picked:
                    adamw_params.append(p)
                    picked.add(id(p))

    # 2) Everything not yet assigned → AdamW (convs, custom tables, etc.)
    for name, p in model.named_parameters():
        if p.requires_grad and id(p) not in picked:
            adamw_params.append(p)

    return muon_params, adamw_params



def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_all_paths(config, wb_run_name=None):
    pred_dens_path_val = config['pred_dens_path_val']
    pred_dens_path_test = config['pred_dens_path_test']
    density_delta_path_val = config['density_delta_path_val']
    density_delta_path_test = config['density_delta_path_test']
    gaus_pos_path_val = config['gaus_pos_path_val']
    gaus_pos_path_test = config['gaus_pos_path_test']
    gaus_pos_outlier_path = config['gaus_pos_outlier_path']
    if config['cluster']:
        pred_dens_path_val = config['cluster_base_path'] + f"split_{config['data_split']}/" + pred_dens_path_val
        pred_dens_path_test = config['cluster_base_path'] + f"split_{config['data_split']}/" + pred_dens_path_test
        density_delta_path_val = config['cluster_base_path'] + f"split_{config['data_split']}/" + density_delta_path_val
        density_delta_path_test = config['cluster_base_path'] + f"split_{config['data_split']}/" + density_delta_path_test
        gaus_pos_path_val = config['cluster_base_path'] + f"split_{config['data_split']}/" + config['gaus_pos_path_val']
        gaus_pos_path_test = config['cluster_base_path'] + f"split_{config['data_split']}/" + config['gaus_pos_path_test']
        gaus_pos_outlier_path = config['cluster_base_path'] + f"split_{config['data_split']}/" + config['gaus_pos_outlier_path']
    if config['wandb']:
        pred_dens_path_val = pred_dens_path_val + f"/{wb_run_name}/"
        pred_dens_path_test = pred_dens_path_test + f"/{wb_run_name}/"
        density_delta_path_val = density_delta_path_val + f"/{wb_run_name}/"
        density_delta_path_test = density_delta_path_test + f"/{wb_run_name}/"
        gaus_pos_path_val = gaus_pos_path_val + f"/{wb_run_name}/"
        gaus_pos_path_test = gaus_pos_path_test + f"/{wb_run_name}/"
        gaus_pos_outlier_path = gaus_pos_outlier_path + f"/{wb_run_name}/"
    # Base save path for all model outputs (e.g. a separate drive). Defaults to the
    # local folder so existing behaviour is preserved.
    output_base_path = config.get('output_base_path', '.')
    pred_dens_path_val = os.path.join(output_base_path, pred_dens_path_val)
    pred_dens_path_test = os.path.join(output_base_path, pred_dens_path_test)
    density_delta_path_val = os.path.join(output_base_path, density_delta_path_val)
    density_delta_path_test = os.path.join(output_base_path, density_delta_path_test)
    gaus_pos_path_val = os.path.join(output_base_path, gaus_pos_path_val)
    gaus_pos_path_test = os.path.join(output_base_path, gaus_pos_path_test)
    gaus_pos_outlier_path = os.path.join(output_base_path, gaus_pos_outlier_path)
    config['pred_dens_path_val'] = pred_dens_path_val
    config['pred_dens_path_test'] = pred_dens_path_test
    config['density_delta_path_val'] = density_delta_path_val
    config['density_delta_path_test'] = density_delta_path_test
    config['gaus_pos_path_val'] = gaus_pos_path_val
    config['gaus_pos_path_test'] = gaus_pos_path_test
    config['gaus_pos_outlier_path'] = gaus_pos_outlier_path
    os.makedirs(pred_dens_path_val, exist_ok=True)
    os.makedirs(pred_dens_path_test, exist_ok=True)
    os.makedirs(density_delta_path_val, exist_ok=True)
    os.makedirs(density_delta_path_test, exist_ok=True)
    os.makedirs(gaus_pos_path_val, exist_ok=True)
    os.makedirs(gaus_pos_path_test, exist_ok=True)
    os.makedirs(gaus_pos_outlier_path, exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)
    return config

def get_files(config):
    file_split = config['data_split']
    file_split_path = resolve_data_path(config['data_split_path'] + f"datasplits_{file_split}.json", "spin_electrafi")
    f = open(file_split_path)
    data = json.load(f)
    train_files_indices = data['train']
    test_files_indices = data['test']
    validation_files_indices = data.get('validation') or data.get('val')
    if file_split in ("gpaw_10k", "mpfull2025_1k", "mpfull2025_10k", "mpfull2025_50k"):
        train_files = [f"{config['dens_path']}/{num}.chgcar.lz4" for num in train_files_indices]
        test_files = [f"{config['dens_path']}/{num}.chgcar.lz4" for num in test_files_indices]
        validation_files = [f"{config['dens_path']}/{num}.chgcar.lz4" for num in validation_files_indices]
    elif file_split not in ("ECD", "ECD_test", "nmc", "mp_mixed", "mpfull2025", "mpfull2025_local"):
        train_files = [f"{config['dens_path']}/{num:06}.CHGCAR.lz4" for num in train_files_indices]
        test_files = [f"{config['dens_path']}/{num:06}.CHGCAR.lz4" for num in test_files_indices]
        validation_files = [f"{config['dens_path']}/{num:06}.CHGCAR.lz4" for num in validation_files_indices]
    elif file_split in ("ECD", "ECD_test"):
        train_files = [f"{config['dens_path']}/{num}.chgcar.lz4" for num in train_files_indices]
        test_files = [f"{config['dens_path']}/{num}.chgcar.lz4" for num in test_files_indices]
        validation_files = [f"{config['dens_path']}/{num}.chgcar.lz4" for num in validation_files_indices]
    elif file_split == "mpfull2025_local":
        # Return bare filenames; data_utils joins dens_path itself
        train_files = [f"{num}.chgcar.lz4" for num in train_files_indices]
        test_files = [f"{num}.chgcar.lz4" for num in test_files_indices]
        validation_files = [f"{num}.chgcar.lz4" for num in validation_files_indices]
    elif file_split == "mp_mixed":
        train_files = [f"{config['dens_path']}/{num:06}.chgcar.lz4" for num in train_files_indices]
        test_files = [f"{config['dens_path']}/{num:06}.chgcar.lz4" for num in test_files_indices]
        validation_files = [f"{config['dens_path']}/{num:06}.chgcar.lz4" for num in validation_files_indices]
    elif file_split == "mpfull2025":
        train_files = [f"{config['dens_path']}/{num}.chgcar.lz4" for num in train_files_indices]
        test_files = [f"{config['dens_path']}/{num}.chgcar.lz4" for num in test_files_indices]
        validation_files = [f"{config['dens_path']}/{num}.chgcar.lz4" for num in validation_files_indices]
    else:
        train_files = [f"{config['dens_path']}/{num}.CHGCAR.lz4" for num in train_files_indices]
        test_files = [f"{config['dens_path']}/{num}.CHGCAR.lz4" for num in test_files_indices]
        validation_files = [f"{config['dens_path']}/{num}.CHGCAR.lz4" for num in validation_files_indices]
    random.shuffle(train_files)
    random.shuffle(test_files)
    random.shuffle(validation_files)

    return train_files, test_files, validation_files

def load_csv_to_dict(file_path, key_column, value_column):
    result_dict = {}
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = row[key_column]
            value = row[value_column]
            result_dict[key] = value
    return result_dict


def get_files_ood(config):
    ood_file_dict = {}
    ood_names = config['ood_names']
    ood_paths = config['ood_paths']
    for name, path in zip(ood_names, ood_paths):
        ood_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.lz4')]
        random.shuffle(ood_files)
        ood_file_dict[name] = {'files': ood_files, 'path': path}

    return ood_file_dict