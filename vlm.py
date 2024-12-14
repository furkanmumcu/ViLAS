import torch
import torch.nn as nn
import timm
from transformers import DistilBertModel, DistilBertConfig
import numpy as np

torch.set_printoptions(sci_mode=False)


class ImageEncoder(nn.Module):
    def __init__(
        self, trainable=False
    ):
        super().__init__()
        self.model = timm.create_model('resnet50', pretrained=False, num_classes=43, global_pool="avg")
        if trainable:
            for p in self.model.parameters():
                p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):
    def __init__(self, pretrained=True, trainable=False):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim=256,
            dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class VLM(nn.Module):
    def __init__(
        self,
        temperature=1.0,
        image_embedding=43,
        text_embedding=768,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        targets = torch.from_numpy(np.arange(text_embeddings.shape[0])).to('cuda')

        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        logits_1 = text_embeddings @ image_embeddings.T
        logits_2 = logits_1.T

        loss_i = nn.CrossEntropyLoss()(logits_1, targets)
        loss_t = nn.CrossEntropyLoss()(logits_2, targets)
        return (loss_i + loss_t) / 2


def cross_entropy(preds, targets, reduction='none', dim=-1):
    log_softmax = nn.LogSoftmax(dim=dim)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

