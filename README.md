# Code for instruction tuning using sentimental adjective-based instance

## Data
Please refer to [data repo](https://github.com/AnonymousGitRepository/data) to download training data and follow the instruction to prepare training and development data. Sample format is [here](https://github.com/AnonymousGitRepository/code/blob/main/ft_datasets/train_and_val_240_word_with_negation_sample.json).

## Prerequisite
Install the necessary packages to run training scripts. Note that instruction tuning requires the `peft` package to enable Low-Rank Adaptation (LoRA) and at least one A100 GPU is needed. 

```python 
pip install -r requirements.txt
```

## Training 
```shell
CUDA_VISIBLE_DEVICES=0 python train.py 
                --model_name_or_path   # llama2 base model or falcon model
                --lora_rank            # for rank in lora
                --lora_alpha           # for alpha in lora
                
```
Note: for llama2 and falcon models, different params will be updated during training by lora approach.
