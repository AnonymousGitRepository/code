#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch, os, random, argparse, sys
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from utils.dataset_utils import get_preprocessed_dataset
from configs.datasets import  alpaca_dataset
from contextlib import nullcontext
from transformers import default_data_collator, Trainer, TrainingArguments


def read_json(path):
    import json
    with open(path, "r") as f:
        data = json.load(f)
    return data


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  #if you are using multi-GPU
    np.random.seed(seed)  #numpy module
    random.seed(seed)  #python random modul
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
def args_init():
    parser=argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name_or_path', type=str, default='meta-llama/Llama-2-7b')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--gradient_checkpointing', type=bool, default=False)
    parser.add_argument('--logging_steps',type=int, default=1)
    parser.add_argument('--save_steps', type=int, default=10)
    parser.add_argument('--eval_steps', type=int, default=10)
    return parser.parse_args()
    

def load_tokenizer(args):
    if 'llama' in args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, padding_side="left", truncation_side="left", model_max_length=256)
        model =LlamaForCausalLM.from_pretrained(args.model_name_or_path, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
        tokenizer.pad_token = tokenizer.bos_token
        model.config.pad_token_id = model.config.bos_token_id   
    elif 'falcon' in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left", truncation_side="left", trust_remote_code=True,return_token_type_ids=False, model_max_length=256)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model =AutoModelForCausalLM.from_pretrained(args.model_name_or_path, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
    return tokenizer, model


def load_dataset(tokenizer):
    train_dataset = get_preprocessed_dataset(tokenizer, alpaca_dataset, 'train')
    val_dataset = get_preprocessed_dataset(tokenizer, alpaca_dataset, 'val')
    return train_dataset, val_dataset


def create_peft_config(args, model, llama2=True):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_int8_training,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules = ["q_proj", "v_proj"] if llama2 else ["query_key_value"]
    )

    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config


if __name__=='__main__':
    args = args_init()
    set_seed(args.seed)
    model, tokenizer = load_tokenizer(args)
    model, lora_config = create_peft_config(model)
    model.train()
    
    train_dataset, val_dataset = load_dataset(tokenizer)

    config = {
        'lora_config': lora_config,
        'learning_rate': args.lr,
        'num_train_epochs': args.num_train_epochs,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'per_device_train_batch_size': args.per_device_train_batch_size,
        'gradient_checkpointing': args.gradient_checkpointing,
    }

    profiler = nullcontext()
    # Define training args
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        overwrite_output_dir=True,
        bf16=True,  # Use BF16 if available
        # logging strategies
        logging_dir=f"{args.out_dir}/logs",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        evaluation_strategy = 'steps',
        save_steps = args.save_steps,
        eval_steps =args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        #optim="adamw_torch_fused",
        max_steps=-1,
        **{k:v for k,v in config.items() if k != 'lora_config'}
    )
    
    with profiler:
        # Create Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=default_data_collator,
            callbacks=[],
            #compute_metrics=compute_metrics_sentiment
        )
    
        # Start training
        trainer.train()

model.save_pretrained(args.out_dir)
print(f"training of {args.model_name_or_path} is done!")
print(f"model has been saved in {args.out_dir}")
