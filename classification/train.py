import os
import json
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from classification.config import (
    MODEL_PATH,
    DATA_PATH,
    OUTPUT_DIR,
    MAX_LENGTH,
    RANDOM_SEED,
    BATCH_SIZE,
    EVAL_BATCH_SIZE,
    NUM_EPOCHS,
    ROBERTA_LR,
    HEAD_LR,
    WEIGHT_DECAY,
    DROPOUT,
    CONV_KERNEL_SIZES,
    NUM_FILTERS,
    USE_BATCH_NORM,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    DEVICE,
    LABEL_NAMES,
)
from classification.dataset import TextDataset
from classification.model import RobertaTextCNN
from classification.evaluate import evaluate
from utils.seed import set_seed


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


def build_optimizer(model):
    param_groups = [
        {
            "params": model.roberta.parameters(),
            "lr": ROBERTA_LR,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": model.convs.parameters(),
            "lr": HEAD_LR,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": model.classifier.parameters(),
            "lr": HEAD_LR,
            "weight_decay": WEIGHT_DECAY,
        },
    ]

    if USE_BATCH_NORM:
        param_groups.append({
            "params": model.bns.parameters(),
            "lr": HEAD_LR,
            "weight_decay": WEIGHT_DECAY,
        })

    return torch.optim.AdamW(param_groups)


def main():
    set_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read data
    df = pd.read_csv(DATA_PATH)

    required_cols = ["label", "text"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].fillna("UNKNOWN").astype(str).str.strip()

    print("总样本数:", len(df))
    print("标签分布:")
    print(df["label"].value_counts())

    # Label encoding
    missing_labels = set(LABEL_NAMES) - set(df["label"].unique().tolist())
    if missing_labels:
        raise ValueError(f"数据中缺少这些标签: {missing_labels}")

    extra_labels = set(df["label"].unique().tolist()) - set(LABEL_NAMES)
    if extra_labels:
        print("警告：数据中存在未纳入训练的额外标签，将被丢弃：", sorted(extra_labels))
        df = df[df["label"].isin(LABEL_NAMES)].copy()

    label2id = {name: i for i, name in enumerate(LABEL_NAMES)}
    id2label = {i: name for name, i in label2id.items()}

    df["label_id"] = df["label"].map(label2id).astype(int)

    with open(os.path.join(OUTPUT_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Split dataset
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

    print("训练集大小:", len(train_df))
    print("验证集大小:", len(valid_df))
    print("测试集大小:", len(test_df))

    train_df["label"].value_counts().to_csv(
        os.path.join(OUTPUT_DIR, "train_label_distribution.csv"),
        encoding="utf-8-sig",
    )
    valid_df["label"].value_counts().to_csv(
        os.path.join(OUTPUT_DIR, "valid_label_distribution.csv"),
        encoding="utf-8-sig",
    )
    test_df["label"].value_counts().to_csv(
        os.path.join(OUTPUT_DIR, "test_label_distribution.csv"),
        encoding="utf-8-sig",
    )

    # Tokenizer and dataloader
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    train_dataset = TextDataset(train_df, tokenizer, MAX_LENGTH)
    valid_dataset = TextDataset(valid_df, tokenizer, MAX_LENGTH)
    test_dataset = TextDataset(test_df, tokenizer, MAX_LENGTH)

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
    )

    # Build model
    model = RobertaTextCNN(
        model_path=MODEL_PATH,
        num_labels=len(LABEL_NAMES),
        num_filters=NUM_FILTERS,
        kernel_sizes=CONV_KERNEL_SIZES,
        dropout=DROPOUT,
        use_batch_norm=USE_BATCH_NORM,
    ).to(DEVICE)

    optimizer = build_optimizer(model)

    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss()

    # Train
    best_macro_f1 = -1.0
    best_epoch = -1
    early_stop_counter = 0
    history = []

    for epoch in range(NUM_EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{NUM_EPOCHS} =====")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            DEVICE,
        )

        valid_loss, valid_metrics, _, _ = evaluate(
            model,
            valid_loader,
            criterion,
            DEVICE,
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")
        print("Valid Metrics:", valid_metrics)

        current_lrs = [group["lr"] for group in optimizer.param_groups]
        print("Current learning rates:", current_lrs)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "roberta_lr_current": current_lrs[0],
            "conv_lr_current": current_lrs[1],
            "classifier_lr_current": current_lrs[2],
            "bn_lr_current": current_lrs[3] if USE_BATCH_NORM else None,
            **valid_metrics,
        })

        current_macro_f1 = valid_metrics["macro_f1"]

        if current_macro_f1 > best_macro_f1 + EARLY_STOPPING_MIN_DELTA:
            best_macro_f1 = current_macro_f1
            best_epoch = epoch + 1
            early_stop_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(OUTPUT_DIR, "best_model.pt"),
            )
            tokenizer.save_pretrained(OUTPUT_DIR)

            with open(os.path.join(OUTPUT_DIR, "best_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_epoch": best_epoch,
                        "best_macro_f1": best_macro_f1,
                        **valid_metrics,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(f"发现更优模型，已保存到 {OUTPUT_DIR}/best_model.pt")
        else:
            early_stop_counter += 1
            print(
                f"验证集 macro_f1 连续 {early_stop_counter} 轮未提升 "
                f"(patience={EARLY_STOPPING_PATIENCE})"
            )

            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                print("触发早停，结束训练。")
                break

    print("\n训练完成")
    print("最佳 epoch:", best_epoch)
    print("最佳 macro_f1:", best_macro_f1)

    # =========================
    # 7. Test
    # =========================
    model.load_state_dict(
        torch.load(
            os.path.join(OUTPUT_DIR, "best_model.pt"),
            map_location=DEVICE,
        )
    )
    model.to(DEVICE)

    test_loss, test_metrics, true_labels, pred_labels = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE,
    )

    print("\n===== 最终测试集整体结果 =====")
    print(test_metrics)

    with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    report_text = classification_report(
        true_labels,
        pred_labels,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )

    print("\n===== 各类别分类结果 =====")
    print(report_text)

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    report_dict = classification_report(
        true_labels,
        pred_labels,
        target_names=LABEL_NAMES,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    with open(os.path.join(OUTPUT_DIR, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(true_labels, pred_labels)
    cm_df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
    cm_df.to_csv(
        os.path.join(OUTPUT_DIR, "confusion_matrix.csv"),
        encoding="utf-8-sig",
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        os.path.join(OUTPUT_DIR, "train_history.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    model_config = {
        "model_name": "roberta_textcnn",
        "pretrained_model_path": MODEL_PATH,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "roberta_lr": ROBERTA_LR,
        "head_lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY,
        "warmup_steps": warmup_steps,
        "dropout": DROPOUT,
        "num_filters": NUM_FILTERS,
        "conv_kernel_sizes": CONV_KERNEL_SIZES,
        "use_batch_norm": USE_BATCH_NORM,
        "num_labels": len(LABEL_NAMES),
        "label2id": label2id,
    }

    with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)

    print(f"\n所有结果已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()