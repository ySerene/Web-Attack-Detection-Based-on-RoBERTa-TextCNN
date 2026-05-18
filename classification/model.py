import torch
import torch.nn as nn
from transformers import AutoModel

class RobertaTextCNN(nn.Module):
    def __init__(
        self,
        model_path,
        num_labels,
        num_filters=128,
        kernel_sizes=(3, 4, 5),
        dropout=0.3,
        use_batch_norm=True,
    ):
        super().__init__()

        self.roberta = AutoModel.from_pretrained(model_path, local_files_only=True)
        hidden_size = self.roberta.config.hidden_size

        self.kernel_sizes = kernel_sizes
        self.use_batch_norm = use_batch_norm

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=num_filters,
                kernel_size=k,
            )
            for k in kernel_sizes
        ])

        if self.use_batch_norm:
            self.bns = nn.ModuleList([
                nn.BatchNorm1d(num_filters)
                for _ in kernel_sizes
            ])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        last_hidden_state = outputs.last_hidden_state  # [B, L, H]

        x = last_hidden_state.transpose(1, 2)  # [B, H, L]
        mask = attention_mask.float()          # [B, L]

        conv_outputs = []

        for idx, (conv, k) in enumerate(zip(self.convs, self.kernel_sizes)):
            c = conv(x)  # [B, F, L-k+1]

            if self.use_batch_norm:
                c = self.bns[idx](c)

            c = torch.relu(c)

            valid_len = c.size(2)
            conv_mask = []

            for i in range(valid_len):
                window = mask[:, i:i + k]
                valid = (window.sum(dim=1) == k).float()
                conv_mask.append(valid)

            conv_mask = torch.stack(conv_mask, dim=1).unsqueeze(1)

            c = c.masked_fill(conv_mask == 0, -1e4)
            pooled = torch.max(c, dim=2)[0]

            conv_outputs.append(pooled)

        feat = torch.cat(conv_outputs, dim=1)
        feat = self.dropout(feat)
        logits = self.classifier(feat)

        return logits