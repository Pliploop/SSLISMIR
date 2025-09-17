from typing import Any

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_info
import importlib
from utils.utils import instantiate
from torch.utils.data import Dataset
from typing import Callable

from utils.utils import load_weights_from_url

from utils.utils import (
    instantiate_from_mapping,
    parse_config,
    MAPPING_NAME_KEY,
    MAPPING_KWARGS_KEY,
)

from torch.utils.data import DataLoader

def get_dataset(config: dict[str, Any], key: str) -> DataLoader:
    dataset = config.get(key, None)
    if dataset is None:
        return None
    dataset = instantiate_from_mapping(dataset)
    return dataset

def get_dataloader_collate_fn(dataset: Dataset, config: dict[str, Any], key: str) -> Callable:
    loader = config.get(key, None)
    if loader is None:
        return None
    collate_fn = loader.pop("collate_fn", None)
    print(collate_fn)
    print(loader)
    collate_fn = instantiate_from_mapping(collate_fn)
    return DataLoader(dataset, **loader, collate_fn=collate_fn)
    

def build_dataloader(config: dict[str, Any]) -> dict[str, DataLoader]:
    dataset_config = config.get("dataset", None)
    train_dataset = get_dataset(dataset_config, "train")
    val_dataset = get_dataset(dataset_config, "val")
    test_dataset = get_dataset(dataset_config, "test")

    
    dataloader_config = config.get('dataloader', None)
    train_loader = get_dataloader_collate_fn(train_dataset, dataloader_config, "train")
    val_loader = get_dataloader_collate_fn(val_dataset, dataloader_config, "val")
    test_loader = get_dataloader_collate_fn(test_dataset, dataloader_config, "test")

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }


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
        validation_loaders = loaders.get("val")
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
