from dataclasses import dataclass

@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/train_and_val_240_word_with_negation_sample.json"
