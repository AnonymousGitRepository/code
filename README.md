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
                --lora_dropout         # for dropout in lora
                --seed                 # random seed   
                --lr                   # learning rate
                --out_dir              # output directory
                --num_train_epochs     # number of epochs
                --gradient_accumulation_steps 
                --per_device_train_batch_size
                --logging_steps # logging steps in trainer
                --save_steps    # checkpoint save steps in trainer
                --eval_steps    # evaluation steps in trainer
                
```
The LoRA model will be saved in the output directory. Note: For Llama2 and Falcon models, different parameters will be updated during training using the LoRA (Low-Rank Adaptation) approach.
