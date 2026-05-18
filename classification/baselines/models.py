import torch
import torch.nn as nn
from transformers import AutoModel

class TextCNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_filters,
        kernel_sizes,
        num_labels,
        dropout,
        pad_id,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_id,
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k,
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_labels)

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids)      # [B, L, E]
        emb = emb.permute(0, 2, 1)           # [B, E, L]

        conv_outputs = []

        for conv in self.convs:
            x = conv(emb)
            x = torch.relu(x)
            x = torch.max(x, dim=2).values
            conv_outputs.append(x)

        feat = torch.cat(conv_outputs, dim=1)
        feat = self.dropout(feat)
        logits = self.classifier(feat)

        return logits

class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_size,
        num_layers,
        num_labels,
        dropout,
        pad_id,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_id,
        )

        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        emb = self.embedding(input_ids)
        outputs, _ = self.bilstm(emb)

        lengths = attention_mask.sum(dim=1) - 1
        lengths = torch.clamp(lengths, min=0)

        batch_indices = torch.arange(outputs.size(0)).to(outputs.device)
        feat = outputs[batch_indices, lengths, :]

        feat = self.dropout(feat)
        logits = self.classifier(feat)

        return logits

class TransformerClassifier(nn.Module):
    def __init__(self, model_path, num_labels, dropout=0.3, local_files_only=False):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        )

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_feat = outputs.last_hidden_state[:, 0, :]
        feat = self.dropout(cls_feat)
        logits = self.classifier(feat)

        return logits