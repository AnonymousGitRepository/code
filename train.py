#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import random
import torch
import numpy as np
from pathlib import Path
from transformers import (
    LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, 
    Trainer, TrainingArguments, default_data_collator
)
from utils.dataset_utils import get_preprocessed_dataset
from configs.datasets import alpaca_dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training

# Constants
SEED = 42
MODEL_PATH = "YOUR_MODEL_PATH_HERE"
LORA_RANK = 8
LORA_ALPHA = 32


def read_json(path):
    """Read JSON file from a given path."""
    with open(path, "r") as f:
        return json.load(f)


def set_seed(seed):
    """Set the seed for all random number generators."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_peft_config(model, rank, alpha):
    """Create PEFT configuration."""
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules = ["q_proj", "v_proj"], #for llama2 model
        target_modules=["query_key_value"] #for falcon model
    )
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    return model, peft_config


def init_model_and_tokenizer(path):
    """Initialize model and tokenizer based on path."""
    if 'llama' in path:
        tokenizer = LlamaTokenizer.from_pretrained(path, padding_side="left", truncation_side="left", model_max_length=256)
        model = LlamaForCausalLM.from_pretrained(path, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
        tokenizer.pad_token = tokenizer.bos_token
        model.config.pad_token_id = model.config.bos_token_id
    elif 'falcon' in path:
        tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left", truncation_side="left", trust_remote_code=True, return_token_type_ids=False, model_max_length=256)
        model = AutoModelForCausalLM.from_pretrained(path, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
    return model, tokenizer


def main():
    set_seed(SEED)

    model, tokenizer = init_model_and_tokenizer(MODEL_PATH)

    train_dataset = get_preprocessed_dataset(tokenizer, alpaca_dataset, 'train')
    val_dataset = get_preprocessed_dataset(tokenizer, alpaca_dataset, 'val')

    model.train()
    model, lora_config = create_peft_config(model, LORA_RANK, LORA_ALPHA)

    # Training configuration
    training_args = TrainingArguments(
        output_dir="output_directory",  # Define your output directory
        overwrite_output_dir=True,
        bf16=True,  # Use BF16 if available
        # Other configurations...
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        # Additional parameters...
    )

    # Start training
    trainer.train()

    # Save and evaluate model
    model.save_pretrained("output_directory")
    model.eval()
    model_input = tokenizer(EVAL_PROMPT, returnensors="pt", return_token_type_ids=False)

    with torch.no_grad():
        print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

    print(f"Training of {MODEL_PATH} is done!")
    print(f"Model has been saved in output_directory")


if __name__ == "__main__":
    main()