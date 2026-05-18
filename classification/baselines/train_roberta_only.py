import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from classification.common.labels import LABEL_NAMES
from classification.common.data_utils import load_and_split_data
from classification.common.transformer_dataset import TransformerTextDataset
from classification.common.train_utils import set_seed, train_with_early_stopping, save_test_results
from classification.baselines.models import TransformerClassifier

MODEL_PATH = "roberta-base"
DATA_PATH = "data/processed/merged_two.csv"
OUTPUT_DIR = "outputs/baselines/roberta_only"

MAX_LENGTH = 256
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 2e-5
DROPOUT = 0.3

PATIENCE = 2
RANDOM_SEED = 3407
LOCAL_FILES_ONLY = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(RANDOM_SEED)

    train_df, valid_df, test_df, label2id, _ = load_and_split_data(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        label_names=LABEL_NAMES,
        random_seed=RANDOM_SEED,
        stratify=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=LOCAL_FILES_ONLY,
    )

    train_dataset = TransformerTextDataset(train_df, tokenizer, MAX_LENGTH)
    valid_dataset = TransformerTextDataset(valid_df, tokenizer, MAX_LENGTH)
    test_dataset = TransformerTextDataset(test_df, tokenizer, MAX_LENGTH)

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    model = TransformerClassifier(
        model_path=MODEL_PATH,
        num_labels=len(LABEL_NAMES),
        dropout=DROPOUT,
        local_files_only=LOCAL_FILES_ONLY,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    history, best_epoch, best_macro_f1 = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        output_dir=OUTPUT_DIR,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        tokenizer=tokenizer,
    )

    model_config = {
        "model_name": "roberta_only",
        "pretrained_model_path": MODEL_PATH,
        "max_length": MAX_LENGTH,
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "dropout": DROPOUT,
        "num_labels": len(LABEL_NAMES),
        "label2id": label2id,
        "early_stopping_patience": PATIENCE,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
    }

    save_test_results(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=DEVICE,
        output_dir=OUTPUT_DIR,
        label_names=LABEL_NAMES,
        history=history,
        model_config=model_config,
    )

if __name__ == "__main__":
    main()