import os
import json
import random
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)

def compute_metrics(y_true, y_pred):
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    acc = accuracy_score(y_true, y_pred)

    return {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
    }


def train_one_epoch(model, dataloader, optimizer, criterion, device):
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

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics, all_labels, all_preds


def train_with_early_stopping(
    model,
    train_loader,
    valid_loader,
    optimizer,
    criterion,
    device,
    output_dir,
    num_epochs,
    patience,
    tokenizer=None,
):
    best_macro_f1 = -1.0
    best_epoch = -1
    no_improve_count = 0
    history = []

    init_loss, init_metrics, _, _ = evaluate(
        model=model,
        dataloader=valid_loader,
        criterion=criterion,
        device=device,
    )

    print("\n===== 训练前验证集结果 =====")
    print(f"Init Loss: {init_loss:.4f}")
    print("Init Metrics:", init_metrics)

    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        valid_loss, valid_metrics, _, _ = evaluate(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=device,
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")
        print("Valid Metrics:", valid_metrics)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            **valid_metrics,
        })

        if valid_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = valid_metrics["macro_f1"]
            best_epoch = epoch + 1
            no_improve_count = 0

            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "best_model.pt"),
            )

            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)

            with open(os.path.join(output_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
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

            print(f"发现更优模型，已保存到 {output_dir}/best_model.pt")

        else:
            no_improve_count += 1
            print(f"验证集 macro_f1 未提升，连续 {no_improve_count} 轮未提升")

            if no_improve_count >= patience:
                print(f"触发早停：连续 {patience} 轮验证集 macro_f1 未提升，停止训练")
                break

    print("\n训练完成")
    print("最佳 epoch:", best_epoch)
    print("最佳 macro_f1:", best_macro_f1)

    return history, best_epoch, best_macro_f1

def save_test_results(
    model,
    test_loader,
    criterion,
    device,
    output_dir,
    label_names,
    history,
    model_config,
):
    model.load_state_dict(
        torch.load(
            os.path.join(output_dir, "best_model.pt"),
            map_location=device,
        )
    )
    model.to(device)

    test_loss, test_metrics, true_labels, pred_labels = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\n===== 最终测试集整体结果 =====")
    print(test_metrics)

    with open(os.path.join(output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    report_text = classification_report(
        true_labels,
        pred_labels,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )

    print("\n===== 各类别分类结果 =====")
    print(report_text)

    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    report_dict = classification_report(
        true_labels,
        pred_labels,
        target_names=label_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    with open(os.path.join(output_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(true_labels, pred_labels)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)

    print("\n===== 混淆矩阵 =====")
    print(cm_df)

    cm_df.to_csv(
        os.path.join(output_dir, "confusion_matrix.csv"),
        encoding="utf-8-sig",
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        os.path.join(output_dir, "train_history.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    with open(os.path.join(output_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)

    print(f"\n所有结果已保存到: {output_dir}")