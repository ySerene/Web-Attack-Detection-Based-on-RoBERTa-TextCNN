import os
import json
from collections import Counter

import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def build_char_vocab(train_texts, output_dir):
    counter = Counter()

    for text in train_texts:
        counter.update(list(str(text)))

    itos = [PAD_TOKEN, UNK_TOKEN] + [ch for ch, _ in counter.most_common()]
    stoi = {ch: i for i, ch in enumerate(itos)}

    vocab = {
        "stoi": stoi,
        "itos": itos,
        "pad_id": stoi[PAD_TOKEN],
        "unk_id": stoi[UNK_TOKEN],
    }

    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "stoi": stoi,
                "itos": itos,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("字符级词表大小:", len(stoi))

    return vocab

def encode_text(text, vocab, max_length):
    stoi = vocab["stoi"]
    pad_id = vocab["pad_id"]
    unk_id = vocab["unk_id"]

    ids = [stoi.get(ch, unk_id) for ch in list(str(text))]

    if len(ids) > max_length:
        ids = ids[:max_length]

    attention_mask = [1] * len(ids)

    if len(ids) < max_length:
        pad_len = max_length - len(ids)
        ids = ids + [pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len

    return ids, attention_mask

class CharTextDataset(Dataset):
    def __init__(self, dataframe, vocab, max_length):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label_id"].tolist()
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        input_ids, attention_mask = encode_text(
            text=text,
            vocab=self.vocab,
            max_length=self.max_length,
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }