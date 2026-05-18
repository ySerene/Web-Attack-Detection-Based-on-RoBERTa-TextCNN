import os
import json
import itertools
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from classification.config import (
    MODEL_PATH,
    DATA_PATH,
    MAX_LENGTH,
    RANDOM_SEED,
    BATCH_SIZE,
    EVAL_BATCH_SIZE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    DEVICE,
    LABEL_NAMES,
)
from classification.dataset import TextDataset
from classification.model import RobertaTextCNN
from classification.evaluate import evaluate
from utils.seed import set_seed

TUNING_OUTPUT_DIR = "outputs/tuning/roberta_textcnn"

NUM_EPOCHS = 8

PARAM_GRID = {
    "roberta_lr": [1e-5, 2e-5],
    "head_lr": [1e-4, 5e-4],
    "dropout": [0.3, 0.4],
    "num_filters": [128],
    "kernel_sizes": [[3, 4, 5]],
}

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def build_optimizer(model, roberta_lr, head_lr):
    param_groups = [
        {
            "params": model.roberta.parameters(),
            "lr": roberta_lr,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": model.convs.parameters(),
            "lr": head_lr,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": model.classifier.parameters(),
            "lr": head_lr,
            "weight_decay": WEIGHT_DECAY,
        },
    ]

    if hasattr(model, "bns"):
        param_groups.append({
            "params": model.bns.parameters(),
            "lr": head_lr,
            "weight_decay": WEIGHT_DECAY,
        })

    return torch.optim.AdamW(param_groups)

def prepare_data():
    df = pd.read_csv(DATA_PATH)

    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].fillna("UNKNOWN").astype(str).str.strip()

    df = df[df["label"].isin(LABEL_NAMES)].copy()

    label2id = {name: i for i, name in enumerate(LABEL_NAMES)}
    df["label_id"] = df["label"].map(label2id).astype(int)

    train_valid_df, test_df = train_test_split(
        df,
        test_size=0.1,
        random_state=RANDOM_SEED,
        stratify=df["label_id"],
    )

    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=0.1111111111,
        random_state=RANDOM_SEED,
        stratify=train_valid_df["label_id"],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    train_dataset = TextDataset(train_df, tokenizer, MAX_LENGTH)
    valid_dataset = TextDataset(valid_df, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, valid_loader

def run_one_trial(trial_id, params, train_loader, valid_loader):
    set_seed(RANDOM_SEED)

    trial_dir = os.path.join(TUNING_OUTPUT_DIR, f"trial_{trial_id}")
    os.makedirs(trial_dir, exist_ok=True)

    model = RobertaTextCNN(
        model_path=MODEL_PATH,
        num_labels=len(LABEL_NAMES),
        num_filters=params["num_filters"],
        kernel_sizes=params["kernel_sizes"],
        dropout=params["dropout"],
        use_batch_norm=True,
    ).to(DEVICE)

    optimizer = build_optimizer(
        model=model,
        roberta_lr=params["roberta_lr"],
        head_lr=params["head_lr"],
    )

    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    best_metrics = None
    best_epoch = -1
    early_stop_counter = 0

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=DEVICE,
        )

        valid_loss, valid_metrics, _, _ = evaluate(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=DEVICE,
        )

        print(
            f"Trial {trial_id} | Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f} | "
            f"macro_f1={valid_metrics['macro_f1']:.4f}"
        )

        current_macro_f1 = valid_metrics["macro_f1"]

        if current_macro_f1 > best_macro_f1 + EARLY_STOPPING_MIN_DELTA:
            best_macro_f1 = current_macro_f1
            best_metrics = valid_metrics
            best_epoch = epoch + 1
            early_stop_counter = 0

            torch.save(model.state_dict(), os.path.join(trial_dir, "best_model.pt"))
        else:
            early_stop_counter += 1

            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                break

    result = {
        "trial_id": trial_id,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        **params,
        **best_metrics,
    }

    with open(os.path.join(trial_dir, "trial_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result

def main():
    os.makedirs(TUNING_OUTPUT_DIR, exist_ok=True)

    train_loader, valid_loader = prepare_data()

    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())

    all_results = []

    for trial_id, combo in enumerate(itertools.product(*values), start=1):
        params = dict(zip(keys, combo))

        print("\n" + "=" * 80)
        print(f"Start Trial {trial_id}")
        print(params)
        print("=" * 80)

        result = run_one_trial(
            trial_id=trial_id,
            params=params,
            train_loader=train_loader,
            valid_loader=valid_loader,
        )

        all_results.append(result)

        results_df = pd.DataFrame(all_results)
        results_df.to_csv(
            os.path.join(TUNING_OUTPUT_DIR, "tuning_results.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="macro_f1", ascending=False)

    results_df.to_csv(
        os.path.join(TUNING_OUTPUT_DIR, "tuning_results_sorted.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== 调参完成 =====")
    print(results_df.head())

    best = results_df.iloc[0].to_dict()
    with open(os.path.join(TUNING_OUTPUT_DIR, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print("\n最佳参数已保存到:")
    print(os.path.join(TUNING_OUTPUT_DIR, "best_params.json"))

if __name__ == "__main__":
    main()
