import torch
import torch.nn as nn
import torch.nn.functional as F


class OneLayerRNN(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim,
            hidden_dim,
            classes_amt,
            rnn_dropout=0.0,
            dropout=0.0
        ):
        super(OneLayerRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            embed_dim,
            hidden_dim,
            batch_first=True,
            nonlinearity='tanh',
            dropout=rnn_dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, classes_amt)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)

        out, _ = torch.max(out, dim=1)
        logits = self.fc(self.dropout(out))

        return logits