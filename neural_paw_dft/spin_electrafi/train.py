import argparse
import glob
import os
import warnings
import yaml
import torch
import wandb
from neural_paw_dft._resources import resolve_data_path
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything

from neural_paw_dft.spin_electrafi.utils.model_handling import ModelIO, get_tag
from neural_paw_dft.spin_electrafi.utils.train_helper_funcs import set_all_seeds, set_all_paths, get_files, get_files_ood
from neural_paw_dft.spin_electrafi.model.ELECTRAFI import ELECTRAFI

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    """Parse CLI arguments allowing selective overrides of the YAML config."""
    parser = argparse.ArgumentParser(
        description="Run ELECTRA training with optional config overrides.")

    # Boolean flag for pruning
    parser.set_defaults(prune=None)

    # Float overrides for learning rates
    parser.add_argument("--initial_lr", type=float, default=None,
                        help="Override the initial learning rate.")
    parser.add_argument("--final_lr", type=float, default=None,
                        help="Override the final learning rate.")
    parser.add_argument("--lr_gamma", type=float, default=None,
                        help="Override the lr gamma")

    load_model_group = parser.add_mutually_exclusive_group()
    load_model_group.add_argument("--load_model", dest="load_model", action="store_true",
                              help="Enable model loading (overrides config).")
    load_model_group.add_argument("--no_load_model", dest="load_model", action="store_false",
                              help="Disable model loading (overrides config).")
    parser.set_defaults(load_model=None)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file, overrides auto-detection.")
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Return a new config with CLI overrides applied when provided."""
    if args.prune is not None:
        config["prune"] = args.prune

    if args.initial_lr is not None:
        config["initial_lr"] = args.initial_lr

    if args.final_lr is not None:
        config["final_lr"] = args.final_lr

    if args.lr_gamma is not None:
        config["lr_gamma"] = args.lr_gamma
    if args.load_model is not None:
        config["load_model"] = args.load_model

    return config


def load_base_config(config_path: str = None) -> dict:
    """Load the base YAML configuration depending on GPU availability and flags inside the YAML."""
    if config_path is not None:
        return yaml.safe_load(open(config_path))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        base_path = "/ELECTRAFI"
        config = yaml.safe_load(open(f"{base_path}/hpc_conf.yaml"))
        if config["data_split"] == "mpfull2025":
            config["master_units"] = 2160
            config["gaus_per_electrons"] = 120
            config["cutoff"] = 30
            config["dens_path"] = "/MP_FULL_2025"
            config["project_name"] = "ELECTRAFI-MP_Full"
            config["lr_gamma"] = 0.9
            config["save_memory"] = True
            config['max_time'] = '39:00:00:00'
            config["ood_eval"] = True
    else:
        torch.set_num_threads(1)
        config = yaml.safe_load(open("local_conf.yaml"))
    return config


def run():
    args = parse_args()
    config = load_base_config(args.config)
    config = apply_overrides(config, args)
    if config['backbone'] == 'escaip':
        escaip_cfg_path = config['escaip_cfg_path']
        with open(resolve_data_path(escaip_cfg_path, "spin_electrafi"), "r") as f:
            escaip_cfg = yaml.safe_load(f)
        config['escaip_config'] = escaip_cfg
        if torch.cuda.is_available():
            if config["data_split"] == "ECD":
                config['escaip_config']['model']["backbone"]["max_num_nodes_per_batch"] = 20
                config['escaip_config']['model']["backbone"]["use_compile"] = True
                config['escaip_config']['model']["backbone"]['max_neighbors'] = 200
            elif config["data_split"] == "cubic":
                config['escaip_config']['model']["backbone"]["max_num_nodes_per_batch"] = 64
                config['escaip_config']['model']["backbone"]["use_compile"] = True
                config['escaip_config']['model']["backbone"]['max_neighbors'] = 128
            elif config["data_split"] == "mp_mixed":
                config['escaip_config']['model']["backbone"]["max_num_nodes_per_batch"] = 154
                config['escaip_config']['model']["backbone"]["use_compile"] = True
                config['escaip_config']['model']["backbone"]['max_neighbors'] = 200
            elif str(config["data_split"]).startswith("mpfull2025"):
                config['escaip_config']['model']["backbone"]["max_num_nodes_per_batch"] = 154
                config['escaip_config']['model']["backbone"]["use_compile"] = True
                config['escaip_config']['model']["backbone"]['max_neighbors'] = 300
            elif config["data_split"] == "qm9":
                config['escaip_config']['model']["backbone"]["max_num_nodes_per_batch"] = 35
                config['escaip_config']['model']["backbone"]["use_compile"] = True
                config['escaip_config']['model']["backbone"]['max_neighbors'] = 35

    # Seed alignment
    set_all_seeds(config['seed'])

    # Checkpoint dir keyed on model_name (defaults to data_split) so a requeued job finds it.
    ckpt_name = config.get('model_name') or config['data_split']
    ckpt_dir = os.path.join(config['model_dir'], "checkpoints", str(ckpt_name))
    os.makedirs(ckpt_dir, exist_ok=True)
    resume_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    # Resume from the newest of last.ckpt and Lightning's SIGUSR1 hpc_ckpt_<N>.ckpt (scoped to ckpt_dir).
    candidates = [p for p in [resume_ckpt, *glob.glob(os.path.join(ckpt_dir, "hpc_ckpt_*.ckpt"))]
                  if os.path.exists(p)]
    ckpt_path = max(candidates, key=os.path.getmtime) if candidates else None

    # Stable wandb run id (persisted beside the checkpoint) so restarts continue one run.
    wandb_id = None
    if config.get('wandb', False):
        id_file = os.path.join(ckpt_dir, "wandb_id.txt")
        if os.path.exists(id_file):
            wandb_id = open(id_file).read().strip()
        else:
            wandb_id = wandb.util.generate_id()
            with open(id_file, "w") as f:
                f.write(wandb_id)

    # Rolling last.ckpt every N steps and at epoch end. Under manual optimization only
    # every_n_train_steps fires, and save_top_k must be 1 for save_last to work.
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,
        save_top_k=1,
        every_n_train_steps=500,
    )

    # Permanent archive (never pruned) so a divergence can be rolled back. The filename
    # prefix keeps it from colliding with checkpoint_cb in the shared dirpath.
    archive_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="archive-{epoch}-{step}",
        save_last=False,
        save_top_k=-1,
        every_n_train_steps=40000,
    )

    # Lightning Trainer defaults: auto strategy is fine for single-GPU
    accel = 'auto' if torch.cuda.is_available() else 'cpu'
    devices = 1

    trainer = L.Trainer(
        accelerator=accel,
        devices=devices,
        # Namespace Lightning's SIGUSR1 checkpoints per run.
        default_root_dir=ckpt_dir,
        callbacks=[checkpoint_cb, archive_cb],
        logger=WandbLogger(
            config=config,
            project=config["project_name"],
            log_model=True,
            group=f"Split_{config['data_split']}",
            # Lets A/B arms on the same split be told apart in the wandb UI.
            name=config.get('run_name', None),
            id=wandb_id,
            resume="allow",
        ) if config.get('wandb', False) else None,
        check_val_every_n_epoch=config['eval_every'],
        log_every_n_steps=1,
        max_epochs=config['max_epochs'],
        # Manual optimization (muon_mix) ignores Trainer-level clipping and errors if set;
        # it clips inside training_step instead. Only pass this for the automatic (AdamW) path.
        gradient_clip_val=(config['gradient_clip_value']
                           if config.get('clip_grad', False)
                           and config.get('optimizer', '').lower() != 'muon_mix'
                           else None),
        gradient_clip_algorithm='norm',
        max_time=config['max_time'],
    )

    # WandB naming and paths
    if config.get('wandb', False):
        wb_name = trainer.logger.experiment.name
        tag = get_tag(wb_name)
    else:
        wb_name = None
        tag = get_tag("test")

    config = set_all_paths(config, wb_name)
    with open(".wandbignore", "w") as f:
        f.write(f"{config['model_dir']}/\n*.pth\n")

    train_files, test_files, validation_files = get_files(config)
    if config["ood_eval"]:
        ood_file_dict = get_files_ood(config)
    model_handler = ModelIO(directory=config['model_dir'], tag=tag) if config.get('save_model', False) else None

    electrafi = ELECTRAFI(
        train_files=train_files,
        test_files=test_files,
        validation_files=validation_files,
        model_handler=model_handler,
        config=config,
    )

    if config.get("load_model", False):
        print(f"Loading model from {config['load_model_path']}")
        electrafi.load_state_dict(torch.load(config['load_model_path']))

    # DataLoaders and seed workers
    train_loader = electrafi.train_dataloader()
    val_loader = electrafi.val_dataloader()
    test_loader = electrafi.test_dataloader()
    seed_everything(config['seed'], workers=True)

    # Run training, auto-resuming from last.ckpt if a previous job left one behind.
    trainer.fit(model=electrafi, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

    # Only test/eval once all epochs are done; a time-stopped job just gets requeued.
    if trainer.current_epoch < config['max_epochs']:
        print(f"Stopped early at epoch {trainer.current_epoch}/{config['max_epochs']} "
              f"(max_time reached). Checkpoint at {resume_ckpt}. Requeue to continue.")
        return

    trainer.test(model=electrafi, dataloaders=test_loader, ckpt_path=None)
    if config["ood_eval"]:
        for name in ood_file_dict.keys():
            ood_loader = electrafi.ood_dataloader(files=ood_file_dict[name]['files'], path=ood_file_dict[name]['path'], name=name)
            trainer.test(model=electrafi, dataloaders=ood_loader, ckpt_path=None)

    # Report where the prediction CHGCARs were written so they are easy to find.
    print("\n=== Prediction CHGCAR locations ===")
    print(f"Test predictions:  {os.path.abspath(config['pred_dens_path_test'])}")
    print(f"Val predictions:   {os.path.abspath(config['pred_dens_path_val'])}")
    print("===================================\n")


if __name__ == "__main__":
    run()


