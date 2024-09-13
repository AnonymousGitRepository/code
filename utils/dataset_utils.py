#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from functools import partial
from typing import Optional
from ft_datasets import get_alpaca_dataset

# Preprocessing mappings for different datasets
DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset, max_words=224),
}

def get_dataset_split(split: str, config) -> str:
    """Return the dataset split based on the provided split type."""
    return config.train_split if split == "train" else config.test_split

def get_preprocessed_dataset(tokenizer, dataset_config, split: str = "train") -> torch.utils.data.Dataset:
    """
    Retrieve a preprocessed dataset based on the specified configuration and split.

    Args:
        tokenizer: Tokenizer to be used for dataset preprocessing.
        dataset_config: Configuration for the dataset.
        split (str): Dataset split to be used, default is 'train'.

    Returns:
        torch.utils.data.Dataset: The preprocessed dataset.
    """
    if dataset_config.dataset not in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_dataset_split(split, dataset_config)
    )