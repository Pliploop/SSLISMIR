from typing import Any

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_info
import importlib
from utils import instantiate

from data.dataset import build_dataloader
from pipelines.train_wrapper import TrainingWrapper
from utils import load_weights_from_url

from utils import (
    instantiate_from_mapping,
    parse_config,
    MAPPING_NAME_KEY,
    MAPPING_KWARGS_KEY,
)


def init_trainer(config: dict[str, Any]) -> L.Trainer:
    # init logger
    logger = config.get("logger", None)
    if logger is not None:
        logger = instantiate_from_mapping(logger)
        # only main process should update config
        if isinstance(logger, WandbLogger) and hasattr(
            logger.experiment.config, "update"
        ):
            logger.experiment.config.update(config, allow_val_change=True)
            # sync all metrics to trainer/global_step
            logger.experiment.define_metric("*", step_metric="trainer/global_step")

    # init callbacks
    callbacks = config.get("callbacks", None)
    if callbacks is not None:
        callbacks = [
            instantiate_from_mapping(c) for c in callbacks.values() if c is not None
        ]

    strategy = config.get("strategy", "auto")
    if strategy != "auto":
        strategy = instantiate_from_mapping(strategy)

    # init trainer
    return L.Trainer(
        logger=logger, callbacks=callbacks, strategy=strategy, **config["trainer"]
    )



def main():
    config = parse_config(show=True, return_dict=True)

    rank_zero_info("Setting CUDA flags...")

    rank_zero_info("Initializing Trainer...")
    trainer = init_trainer(config)

    with trainer.init_module():
        model = instantiate(
            config["__wrapper__"],
            **config["model"],
            loss_params=config["loss"],
            opt_params=config["optimizer"],
            sched_params=config.get("scheduler", None),
            ema_params=config.get("ema", None)
        )

    if isinstance(trainer.logger, WandbLogger):
        trainer.logger.watch(model)

    loaders = build_dataloader(config)

    if isinstance(loaders, dict):
        train_loader = loaders.get("train")
        if not train_loader:
            raise ValueError("No training loader found in the configuration.")
        validation_loaders = loaders.get("validation")
        test_loaders = loaders.get("test")
    else:
        train_loader = loaders
        validation_loaders = None
        test_loaders = None


    rank_zero_info("Starting training...")
    try:
        ckpt_path=config.get("ckpt_path", None)
        trainer.fit(model, train_dataloaders=train_loader, 
        val_dataloaders=validation_loaders,
        ckpt_path=ckpt_path)
    except Exception as e:
        rank_zero_info(f"Error during training: {e}")
        if hasattr(trainer, "logger") and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log({"error": str(e)})
        raise e
    finally:
        rank_zero_info("Training finished.")


if __name__ == "__main__":
    main()
