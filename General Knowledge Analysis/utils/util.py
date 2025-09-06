import torch
from torch.utils.data import DataLoader
import datasets

from utils.MyConfig import *
from utils.custom_dataset import CustomDataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

import numpy as np
import random
import os

from typing import (
    Dict,
    Tuple
)

def load_general_dataset(path:str, data_file:Dict) -> datasets.Dataset:
    """
    This function gets a path and data file to return an existing dataset in your disk.
    :param path: The path of dataset.
    :param data_file: The train and test files.
    :return:
    """
    return datasets.load_dataset(path=path, data_files=data_file)

def get_samples(your_dataset:datasets.Dataset, n_samples:int, seed:int=42) -> datasets.Dataset:
    """
    This function will return a subset of a dataset.
    :param your_dataset: Your dataset.
    :param n_samples: Number of samples you need from your dataset.
    :param seed: For reproducibility.
    :return:
    Subset of corresponding dataset.
    """
    return your_dataset.shuffle(seed).select(range(n_samples))

def create_torch_dataset(dataset:datasets.Dataset, tokenizer:AutoTokenizer, use_attention:bool=False) -> CustomDataset:
    """
    This function will create custom dataset compatible with torch dataset.
    :param dataset: The text dataset(it must be for testing purpose.)
    :param tokenizer: The tokenizer to tokenize each input text.
    :param use_attention: If true, use attention mechanism.
    :return:
    CustomDatasetCKA.
    """
    return CustomDataset(
        text_dataset=dataset,
        tokenizer=tokenizer,
        use_attention=use_attention,
    )

def dataloader(compatible_dataset:CustomDataset, generator:torch.Generator, batch:int=8, shuffle:bool=True) -> DataLoader:
    """
    This function will create a data loader.
    :param compatible_dataset: A custom dataset instantiated form class above.
    :param batch: The number of samples in each batch.
    :param shuffle: Shuffling dataset before batching.
    :param generator: The generator for reproducibility
    :return:
    DataLoader.
    """
    return DataLoader(compatible_dataset, batch_size=batch, shuffle=shuffle, generator=generator)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def model_and_tokenizer(model_name: str, local_dir: str ="models") -> Tuple[AutoModelForCausalLM.from_pretrained, AutoTokenizer.from_pretrained]:
    """
    This function will load the quantized version of specified model.
    :param model_name: The model name
    :param local_dir: Directory of the downloaded the model
    :return:
        Model and Tokenizer.
    """
    os.makedirs(local_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side='right', model_max_length=MAX_LENGTH, cache_dir=local_dir
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, quantization_config=bnb_config, cache_dir=local_dir
    )
    return model, tokenizer
