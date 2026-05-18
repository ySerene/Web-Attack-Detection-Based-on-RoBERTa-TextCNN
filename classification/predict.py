import os
import json
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from classification.model import RobertaTextCNN
from utils.utils import normalize_for_roberta

MODEL_DIR = "outputs/classification/roberta_textcnn"
MODEL_PATH = "outputs/dapt_lr_search/roberta_dapt_lr2e-05/final"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
LABEL_MAPPING_PATH = os.path.join(MODEL_DIR, "label_mapping.json")

MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_input_text(method="", uri="", cookie="", referer="", body=""):
    """
    构造与训练阶段一致的模型输入格式。
    """
    method = normalize_for_roberta(method)
    uri = normalize_for_roberta(uri)
    cookie = normalize_for_roberta(cookie)
    referer = normalize_for_roberta(referer)
    body = normalize_for_roberta(body)

    text = (
        f"[METHOD] {method} "
        f"[URI] {uri} "
        f"[COOKIE] {cookie} "
        f"[REFERER] {referer} "
        f"[BODY] {body}"
    )

    return " ".join(text.split())

def load_label_mapping(label_mapping_path):
    with open(label_mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    id2label = mapping["id2label"]
    id2label = {int(k): v for k, v in id2label.items()}

    return id2label

def load_model(model_path, best_model_path, num_labels):
    model = RobertaTextCNN(
        model_path=model_path,
        num_labels=num_labels,
        num_filters=128,
        kernel_sizes=[3, 4, 5],
        dropout=0.3,
        use_batch_norm=True,
    )

    state_dict = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model

@torch.no_grad()
def predict(model, tokenizer, text, id2label):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = F.softmax(logits, dim=1).squeeze(0)

    pred_id = torch.argmax(probs).item()
    pred_label = id2label[pred_id]
    pred_prob = probs[pred_id].item()

    all_probs = {
        id2label[i]: round(probs[i].item(), 6)
        for i in range(len(id2label))
    }

    return pred_label, pred_prob, all_probs

def main():
    parser = argparse.ArgumentParser(description="Use trained RoBERTa-TextCNN model for Web attack detection.")

    parser.add_argument("--method", type=str, default="GET", help="HTTP method")
    parser.add_argument("--uri", type=str, required=True, help="Request URI")
    parser.add_argument("--cookie", type=str, default="", help="Cookie field")
    parser.add_argument("--referer", type=str, default="", help="Referer field")
    parser.add_argument("--body", type=str, default="", help="Request body")

    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="DAPT RoBERTa model path")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR, help="Trained classifier output directory")

    args = parser.parse_args()

    best_model_path = os.path.join(args.model_dir, "best_model.pt")
    label_mapping_path = os.path.join(args.model_dir, "label_mapping.json")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"未找到模型权重文件: {best_model_path}")

    if not os.path.exists(label_mapping_path):
        raise FileNotFoundError(f"未找到标签映射文件: {label_mapping_path}")

    id2label = load_label_mapping(label_mapping_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    model = load_model(
        model_path=args.model_path,
        best_model_path=best_model_path,
        num_labels=len(id2label),
    )

    text = build_input_text(
        method=args.method,
        uri=args.uri,
        cookie=args.cookie,
        referer=args.referer,
        body=args.body,
    )

    pred_label, pred_prob, all_probs = predict(
        model=model,
        tokenizer=tokenizer,
        text=text,
        id2label=id2label,
    )

    print("\n===== 输入文本 =====")
    print(text)

    print("\n===== 检测结果 =====")
    print(f"预测类别: {pred_label}")
    print(f"置信度: {pred_prob:.6f}")

    print("\n===== 各类别概率 =====")
    for label, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"{label}: {prob}")

if __name__ == "__main__":
    main()