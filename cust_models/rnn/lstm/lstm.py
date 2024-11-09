import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchLSTM(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim,
            hidden_dim,
            classes_amt,
            lstm_dropout=0.0,
            dropout=0.0,
            bidirectional=False
        ):
        super(TorchLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)

        d = 2 if bidirectional else 1
        self.fc = nn.Linear(d * hidden_dim, classes_amt)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        out, _ = self.lstm(x)

        out, _ = torch.max(out, dim=1)
        logits = self.fc(self.dropout(out))

        return logits
    

class DisableGateLSTM(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim,
            hidden_dim,
            classes_amt,
            dropout=0.0,
            disable_gate=None
        ):
        super(DisableGateLSTM, self).__init__()

        self.hid_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm_cell = LSTM(embed_dim, hidden_dim, disable_gate)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, classes_amt)
        self.disable_gate = disable_gate
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        batch_size, text_len = x.size(0), x.size(1)

        h_t = torch.zeros(batch_size, self.hid_dim)
        c_t = torch.zeros(batch_size, self.hid_dim)

        hids = []
        for t in range(text_len):
            x_t = x[:, t, :]
            h_t, c_t = self.lstm_cell(x_t, h_t, c_t)
            hids.append(h_t.unsqueeze(1))

        hids = torch.cat(hids, dim=1)
        max_hids, _ = torch.max(hids, dim=1)
        logits = self.fc(self.dropout(max_hids))
        
        return logits


class LSTM(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            disable_gate=None
        ):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.disable_gate = disable_gate
        
        self.W_f = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        self.W_i = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        self.W_o = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        self.W_c = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
    
    def forward(self, x, prev_h, prev_c):
        batch_size = x.size(0)
        input = torch.cat((prev_h, x), 1)

        if self.disable_gate == 'f':
            f = torch.ones((batch_size, self.hidden_dim))
        else:
            f = torch.sigmoid(self.W_f(input))

        if self.disable_gate == 'i':
            i = torch.ones((batch_size, self.hidden_dim))
        else:
            i = torch.sigmoid(self.W_i(input))

        if self.disable_gate == 'o':
            o = torch.ones((batch_size, self.hidden_dim))
        else:
            o = torch.sigmoid(self.W_o(input))

        c_temp = torch.tanh(self.W_c(input))

        c = f * prev_c + i * c_temp
        h = o * torch.tanh(c)
        return h, c
    

class MultilayerLSTM(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim,
            hidden_dim,
            classes_amt,
            layers_amt,
            lstm_dropout=0.0,
            dropout=0.0,
            bidirectional=False
        ):
        super(MultilayerLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=layers_amt,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, classes_amt)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        out, _ = self.lstm(x)

        out, _ = torch.max(out, dim=1)
        logits = self.fc(self.dropout(out))

        return logits