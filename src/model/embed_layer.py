from typing import List

import torch
import torch.nn as nn
import numpy


class EmbedingLayer(nn.Module):

    def __init__(self,
                 num_numerical_fields: int,
                 num_categorical_fields: int,
                 num_ids: List[int],
                 embed_size: int,
                 device: str="cpu"):
        super(EmbedingLayer, self).__init__()

        # map id=0 if id is NULL. that's why adding a aditional dimention.
        if num_numerical_fields>0:
            self.conv = nn.Conv1d(1, embed_size*2, 1)
            self.num_numerical_fields = num_numerical_fields
        self.num_categorical_fields = num_categorical_fields
        self.embed_size = embed_size

        # bind num_field to the length of field_sizes
        self.num_fields = len(num_ids)

        # create ModuleList of nn.Embedding for each field of inputs
        # map id=0 if id is NULL. that's why adding a aditional dimention.        
        self.embeddings = nn.ModuleList([
            nn.Embedding(sum(num_ids)+1, embed_size) for _ in range(self.num_fields)
        ])

        # create offsets to re-index inputs by adding them up
        ## self.offsets = torch.Tensor((0, *np.cumsum(num_ids)[:-1])).long().unsqueeze(0)
        self.offsets = torch.Tensor((0, *numpy.cumsum(num_ids)[:-1])).long()
        self.offsets.names = ("N",)
        self.offsets = self.offsets.unflatten("N", [("B", 1), ("N", self.offsets.size("N"))])
        self.offsets.to(device)

        # initialize nn.Embedding with xavier_uniform_ initializer
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self,
                X_categorical: List[torch.Tensor],
                X_numerical: torch.Tensor = None
        ) -> torch.Tensor:

        X_categorical = X_categorical + self.offsets
        X_categorical = X_categorical.rename(None)

        if X_numerical is None:
            outputs = torch.cat([self.embeddings[i](X_categorical) for i in range(self.num_fields)], dim=1)
            outputs.names = ("B", "N", "E")
            return outputs

        numerical_emb = self.conv(X_numerical).reshape(-1, self.num_numerical_fields*2, self.embed_size)
        cat_emb = torch.cat([self.embeddings[i](X_categorical) for i in range(self.num_fields)], dim=1)
        outputs = torch.cat([numerical_emb, cat_emb], dim=1)
        outputs.names = ("B", "N", "E")
        return outputs