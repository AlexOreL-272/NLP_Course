import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim,
            filters_amt,
            classes_amt,
            filter_widths,
            dropout=0.0
        ):
        super(SimpleCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1d_layers = nn.ModuleList()
        for width in filter_widths:
            self.conv1d_layers.append(
                nn.Conv1d(embed_dim, filters_amt, width)
            )

        self.fc = nn.Linear(len(filter_widths) * filters_amt, classes_amt)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)

        # x shape should be (batch_size, emb_size, text_length)
        x = x.permute(0, 2, 1)

        conv_outputs = []
        for conv1d in self.conv1d_layers:
            conv_out = conv1d(x.float())

            pooled_out = F.max_pool1d(
                conv_out, conv_out.size(2)
            )
            
            pooled_out = pooled_out.squeeze(2)
            conv_outputs.append(pooled_out)

        concat_output = torch.cat(conv_outputs, dim=1)

        output = self.fc(self.dropout(concat_output))
        
        # somehow it doesn't work
        # output = F.softmax(output, dim=1)

        return output
