from typing import Optional, Tuple
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
from pathlib import Path
import torch
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
#from pytorch_lightning.trainingtype import DDPPlugin

from yacs.config import CfgNode
from hamer_uncertainty.configs import dataset_config
from hamer_uncertainty.datasets import HAMERDataModule
from hamer_uncertainty.models.hamer import HAMER
from hamer_uncertainty.models.hamer_diag import HAMER_DIAG
from hamer_uncertainty.models.hamer_full import HAMER_FULL
from hamer_uncertainty.models.hamer_ours import HAMER_OURS
from hamer_uncertainty.models.hamer_ours_wo_linear import HAMER_OURS_WO_LINEAR
from hamer_uncertainty.utils.pylogger import get_pylogger
from hamer_uncertainty.utils.misc import task_wrapper, log_hyperparameters
from pytorch_lightning.strategies.ddp import DDPStrategy
# HACK reset the signal handling so the lightning is free to set it
# Based on https://github.com/facebookincubator/submitit/issues/1709#issuecomment-1246758283
import signal
signal.signal(signal.SIGUSR1, signal.SIG_DFL)

log = get_pylogger(__name__)

@pl.utilities.rank_zero.rank_zero_only
def save_configs(model_cfg: CfgNode, dataset_cfg: CfgNode, rootdir: str):
    """Save config files to rootdir."""
    Path(rootdir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=model_cfg, f=os.path.join(rootdir, 'model_config.yaml'))
    with open(os.path.join(rootdir, 'dataset_config.yaml'), 'w') as f:
        f.write(dataset_cfg.dump())

@task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:

    # Load dataset config
    dataset_cfg = dataset_config()

    # Save configs
    save_configs(cfg, dataset_cfg, cfg.paths.output_dir)

    # Setup training and validation datasets
    datamodule = HAMERDataModule(cfg, dataset_cfg)
    # Setup model
    if cfg.model_type == 'deterministic':
        model = HAMER(cfg)
    elif cfg.model_type == 'full':
        model = HAMER_FULL(cfg)
    elif cfg.model_type == 'ours':
        model = HAMER_OURS(cfg)
    elif cfg.model_type == 'ours_wo_linear':
        model = HAMER_OURS_WO_LINEAR(cfg)
    elif cfg.model_type == 'diag':
        model = HAMER_DIAG(cfg)
    else:
        print(f"There is no model_type: {cfg.model_type}")

    # Setup Tensorboard logger
    logger = TensorBoardLogger(os.path.join(cfg.paths.output_dir, 'tensorboard'), name='', version='', default_hp_metric=False)
    loggers = [logger]

    # Setup checkpoint saving
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.output_dir, 'checkpoints'),
        every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS,
        save_last=True,
        save_top_k=cfg.GENERAL.CHECKPOINT_SAVE_TOP_K,
    )
    rich_callback = pl.callbacks.RichProgressBar()
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks = [
        checkpoint_callback,
        lr_monitor,
        # rich_callback
    ]

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=callbacks,
        logger=loggers,
        #plugins=(SLURMEnvironment(requeue_signal=signal.SIGUSR2) if (cfg.get('launcher',None) is not None) else DDPPlugin(find_unused_parameters=False)), # Submitit uses SIGUSR2
        plugins=(SLURMEnvironment(requeue_signal=signal.SIGUSR2) if (cfg.get('launcher',None) is not None) else None), # Submitit uses SIGUSR2
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # finetuning
    checkpoint = torch.load('./_DATA/hamer_ckpts/checkpoints/hamer.ckpt', map_location=model.device)
    load_state_dict(model, checkpoint['state_dict'])
    #######

    # Train the model
    trainer.fit(model, datamodule=datamodule, ckpt_path='last')
    log.info("Fitting done")


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys
    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

@hydra.main(version_base="1.2", config_path=str(root/"hamer_uncertainty/configs_hydra"), config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
