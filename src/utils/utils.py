import torch.nn as nn
from typing import Any
from copy import deepcopy


def build_model(model_conf: dict[str, Any], show: bool = False) -> nn.Module:
    """
    Build a model from a configuration.

    Args:
        model_conf: Configuration for the model.
        show: Whether to print the model.

    Returns:
        model: The built model.
    """
    # copy the model config since it might be modified inplace
    
    print(f"building model {model_conf.get(MAPPING_NAME_KEY)} with weights {model_conf.get('weights_url')}") 
    model_conf = deepcopy(model_conf)

    model_kwargs = model_conf.get(MAPPING_KWARGS_KEY, {})
    if model_kwargs:
        for k, v in model_kwargs.items():
            if isinstance(v, dict) and v.get(MAPPING_NAME_KEY, None):
                # inplace change
                model_kwargs[k] = build_model(v)

    # instantiate the model
    model = instantiate_from_mapping(model_conf)

    # load the weights if provided
    if model_conf.get("weights_url"):
        weights = load_weights_from_url(
            model_conf["weights_url"], profile=model_conf.get("profile")
        )
        
        if weights.get("state_dict", None):
            # its Lightning checkpoint
            weights = weights["state_dict"]
        elif weights.get("module", None):
            # its DeepSpeed checkpoint
            weights = weights["module"]

        
        if model_conf.get("weights_key"):
            weights_key = model_conf["weights_key"]
            if isinstance(weights_key, str):
                weights_key = [weights_key]

            # Use a single dictionary comprehension to filter and replace keys
            new_weights = {
                k.replace(key, ""): v
                for key in weights_key
                for k, v in weights.items()
                if k.startswith(key)
            }
            weights = new_weights

        
        model.load_state_dict(weights, strict=False)

    
    if model_conf.get("trainable_layers", None) is not None:
        for pattern in model_conf["trainable_layers"]:
            match_modules_to_pattern(model, pattern, exclude=False)

    if model_conf.get("frozen_layers", None) is not None:
        for pattern in model_conf["frozen_layers"]:
            match_modules_to_pattern(model, pattern, exclude=True)
        strict = model_conf.get("strict", True)
        model.load_state_dict(weights, strict=strict)

    # freeze the model if provided
    if model_conf.get("freeze", False):
        rank_zero_info(f"Freezing the {model_conf[MAPPING_NAME_KEY]}")
        _ = model.eval()
        for param in model.parameters():
            param.requires_grad = False



    if show:
        rank_zero_info(model)



    return model

import importlib
import os
import subprocess
from collections import OrderedDict
from functools import partial
from hashlib import sha256
from typing import Any, Callable, Mapping

from lightning.pytorch.utilities.rank_zero import rank_zero_info
from omegaconf import DictConfig, ListConfig, OmegaConf

WEIGHTS_DIR: str = "weights/"
PT_WEIGHTS_EXTENSIONS: tuple[str, ...] = (".pth", ".pt", ".ckpt", ".bin")
SAFETENSORS_EXTENSIONS: tuple[str, ...] = (".safetensors",)
CONFIG_TEMPLATE: str = "src/config/{name}.yaml"
MAPPING_NAME_KEY: str = "_name_"
MAPPING_KWARGS_KEY: str = "_kwargs_"
MAPPING_YAML_KEY: str = "_yaml_"
MAPPING_DEFAULT_KEY: str = "_default_"


def instantiate(name: str, **kwargs) -> Callable:
    """
    Dynamically import and instantiate a function/class from a fully qualified name.

    Args:
        name: Fully qualified function/class name (e.g., 'module.submodule.function')
        kwargs: Optional keyword arguments to partially apply to the function

    Returns:
        The function/class object, optionally partially applied with the provided arguments
    """
    if "." in name:
        module_name, class_or_function_name = name.rsplit(".", 1)
    else:
        raise ValueError(
            f"Function name must be fully qualified (module.function), got {name}"
        )

    module = importlib.import_module(module_name)
    class_or_function = getattr(module, class_or_function_name)

    if isinstance(class_or_function, type):
        obj = class_or_function(**kwargs) if kwargs else class_or_function()
        return obj
    elif callable(class_or_function):
        return partial(class_or_function, **kwargs) if kwargs else class_or_function
    else:
        raise TypeError(f"{name} is neither a class nor a callable function")


def instantiate_from_mapping(mapping: Mapping[str, Any], **kwargs) -> Callable:
    """
    Recursively instantiate a function/class from an arbitrary mapping.
    Handles nested mappings by recursively instantiating them.
    """
    if not mapping.get(MAPPING_NAME_KEY, None):
        raise ValueError(f"{MAPPING_NAME_KEY} is required in the mapping")

    mapping_kwargs = mapping.get(MAPPING_KWARGS_KEY, {})

    processed_kwargs = {}
    for k, v in mapping_kwargs.items():
        if isinstance(v, dict) and MAPPING_NAME_KEY in v:
            processed_kwargs[k] = instantiate_from_mapping(v)
        else:
            processed_kwargs[k] = v

    processed_kwargs.update(kwargs)

    return instantiate(mapping[MAPPING_NAME_KEY], **processed_kwargs)


def load_weights(weights_path: str) -> OrderedDict:
    """Load model weights from a local file.

    Args:
        weights_path: Path to the weights file

    Returns:
        The loaded weights
    """
    if weights_path.endswith(PT_WEIGHTS_EXTENSIONS):
        import torch

        return torch.load(weights_path)
    elif weights_path.endswith(SAFETENSORS_EXTENSIONS):
        from safetensors.torch import load_file

        return load_file(weights_path)
    else:
        raise ValueError(f"Invalid weights file: {weights_path}")



def parse_config(
    show: bool = False, return_dict: bool = False
) -> dict[str, Any] | DictConfig:
    """
    Parse the config file and handle command line overrides.

    Mandatory argument for CLI: config=<name> which specifies the base configuration file to use.

    The function supports several operations:
    1. Loading a base config file specified by the 'config' parameter
    2. Merging default configurations if specified in the base config
    3. Applying command line overrides to any configuration parameter
    4. Removing configuration keys using the '~' parameter (comma-separated paths)

    Example:
        python train.py config=pretrain_emilia other.config.param=value
        python train.py config=pretrain_emilia ~=model.param1,model.param2

    Args:
        show (bool, optional): Whether to print the final configuration. Defaults to False.

    Returns:
        dict[str, Any]: The parsed configuration with all defaults and CLI overrides applied

    Raises:
        ValueError: If no configuration file is provided or if a key to remove is not found
    """
    # get base config name and overrides from CLI
    config_overrides = OmegaConf.from_cli()

    # pop config name, default overrides and remove overrides from CLI
    config_name = config_overrides.pop("config", None)
    default_overrides = config_overrides.pop("defaults", None)
    remove_overrides = config_overrides.pop("~", None)
    other_overrides = config_overrides

    # load base config
    if config_name is None:
        raise ValueError("No configuration file provided")
    config_base = OmegaConf.load(CONFIG_TEMPLATE.format(name=config_name))

    # merge default overrides with base config
    if default_overrides:
        config_base = OmegaConf.merge(config_base, {"defaults": default_overrides})

    # load default configs if they exist
    config_base_defaults = config_base.pop("defaults", None)
    if config_base_defaults:
        default_configs = [
            OmegaConf.load(CONFIG_TEMPLATE.format(name=f"{k}/{name}"))
            for k, name in config_base_defaults.items()
        ]
        config_base = OmegaConf.merge(*default_configs, config_base)

    # merge other overrides with base config
    config = OmegaConf.merge(config_base, other_overrides)

    # remove overrides from config
    if remove_overrides:
        for key_path in remove_overrides.split(","):
            keys = key_path.split(".")
            node = config
            for key in keys[:-1]:
                node = node[key]
            del node[keys[-1]]

    # load yaml references
    config = load_yaml_with_references(config)

    # show config if requested
    if show:
        rank_zero_info(OmegaConf.to_yaml(config, resolve=True))

    if return_dict:
        return OmegaConf.to_container(config, resolve=True)
    return config


def merge_configs(config1: dict[str, Any], config2: dict[str, Any]) -> dict[str, Any]:
    """
    Merge two dictionaries recursively using OmegaConf.
    """
    return OmegaConf.to_container(OmegaConf.merge(config1, config2))
