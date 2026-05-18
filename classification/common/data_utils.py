import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(
    data_path,
    output_dir,
    label_names,
    random_seed=3407,
    test_size=0.1,
    valid_size=0.1111111111,
    stratify=True,
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    required_cols = ["label", "text"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].fillna("UNKNOWN").astype(str).str.strip()

    print("总样本数:", len(df))
    print("标签分布:")
    print(df["label"].value_counts())

    missing_labels = set(label_names) - set(df["label"].unique().tolist())
    if missing_labels:
        raise ValueError(f"数据中缺少这些标签: {missing_labels}")

    extra_labels = set(df["label"].unique().tolist()) - set(label_names)
    if extra_labels:
        print("警告：数据中存在未纳入训练的额外标签，将被丢弃：", sorted(extra_labels))
        df = df[df["label"].isin(label_names)].copy()

    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}

    df["label_id"] = df["label"].map(label2id).astype(int)

    with open(os.path.join(output_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    stratify_col = df["label_id"] if stratify else None

    train_valid_df, test_df = train_test_split(
        df,
        test_size=test_size,
        shuffle=True,
        random_state=random_seed,
        stratify=stratify_col,
    )

    stratify_col_2 = train_valid_df["label_id"] if stratify else None

    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=valid_size,
        shuffle=True,
        random_state=random_seed,
        stratify=stratify_col_2,
    )

    print("训练集大小:", len(train_df))
    print("验证集大小:", len(valid_df))
    print("测试集大小:", len(test_df))

    train_df["label"].value_counts().to_csv(
        os.path.join(output_dir, "train_label_distribution.csv"),
        encoding="utf-8-sig",
    )
    valid_df["label"].value_counts().to_csv(
        os.path.join(output_dir, "valid_label_distribution.csv"),
        encoding="utf-8-sig",
    )
    test_df["label"].value_counts().to_csv(
        os.path.join(output_dir, "test_label_distribution.csv"),
        encoding="utf-8-sig",
    )

    return train_df, valid_df, test_df, label2id, id2label