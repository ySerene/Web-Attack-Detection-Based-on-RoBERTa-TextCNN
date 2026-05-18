import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from classification.common.labels import LABEL_NAMES
from classification.common.data_utils import load_and_split_data
from classification.common.char_dataset import build_char_vocab, CharTextDataset
from classification.common.train_utils import train_with_early_stopping, save_test_results
from classification.baselines.models import TextCNNClassifier

DATA_PATH = "data/processed/merged_two.csv"
OUTPUT_DIR = "outputs/baselines/textcnn_only"

MAX_LENGTH = 256
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
DROPOUT = 0.3

EMBED_DIM = 128
NUM_FILTERS = 128
KERNEL_SIZES = [3, 4, 5]

PATIENCE = 2
RANDOM_SEED = 3407

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df, valid_df, test_df, label2id, _ = load_and_split_data(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        label_names=LABEL_NAMES,
        random_seed=RANDOM_SEED,
        stratify=True,
    )

    vocab = build_char_vocab(
        train_texts=train_df["text"].tolist(),
        output_dir=OUTPUT_DIR,
    )

    train_dataset = CharTextDataset(train_df, vocab, MAX_LENGTH)
    valid_dataset = CharTextDataset(valid_df, vocab, MAX_LENGTH)
    test_dataset = CharTextDataset(test_df, vocab, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    model = TextCNNClassifier(
        vocab_size=len(vocab["stoi"]),
        embed_dim=EMBED_DIM,
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        num_labels=len(LABEL_NAMES),
        dropout=DROPOUT,
        pad_id=vocab["pad_id"],
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
    )

    model_config = {
        "model_name": "textcnn_only",
        "input_type": "char_level",
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "dropout": DROPOUT,
        "embed_dim": EMBED_DIM,
        "num_filters": NUM_FILTERS,
        "kernel_sizes": KERNEL_SIZES,
        "vocab_size": len(vocab["stoi"]),
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