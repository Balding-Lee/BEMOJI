import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    def __init__(self, num_emojis):
        self.batch_size = 16
        self.max_seq_length = 64
        self.lr = 1e-3
        self.epoch = 50
        self.early_stop = 512

        self.num_emojis = num_emojis
        self.word_embedding_size = 300
        self.emoji_embedding_size = 150

        self.lstm_hidden_size = 150
        self.num_layers = 1
        self.num_outputs = 2


class Model(nn.Module):
    def __init__(self, embed, config, device):
        super().__init__()
        self.device = device

        self.word_embeddings = nn.Embedding.from_pretrained(embed, freeze=False)
        self.pos_emoji_embedding = nn.Embedding(config.num_emojis,
                                                config.emoji_embedding_size)
        self.neg_emoji_embedding = nn.Embedding(config.num_emojis,
                                                config.emoji_embedding_size)

        self.attention_bias = nn.Parameter(torch.zeros(config.max_seq_length, ))
        self.relu = nn.ReLU()
        self.LSTM = nn.LSTM(config.word_embedding_size + config.emoji_embedding_size * 2,
                            config.lstm_hidden_size, num_layers=config.num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(config.lstm_hidden_size * 2, config.num_outputs)
        self.sigmoid = nn.Sigmoid()

    def attention(self, e, x):
        """
        f_att function
        :param e: tensor
                shape: (batch, 1, emoji_embed_size * 2)
                emoji embeds
        :param x: tensor
                shape: (batch, max_seq_length, word_embed_size)
                sequence embeds
        :return vt: tensor
                shape: (batch, max_seq_length, emoji_embed_size * 2)
        """
        # shape: (batch, 1, max_seq_length)
        u = self.relu(torch.matmul(e, x.permute(0, 2, 1)) + self.attention_bias)
        alpha = F.softmax(u)
        # shape: (batch, max_seq_length, emoji_embed_size * 2)
        vt = torch.matmul(alpha.permute(0, 2, 1), e)

        return vt

    def forward(self, input_seqs, input_emojis):
        input_seqs = torch.LongTensor(input_seqs).to(self.device)
        # shape: (batch_size, max_seq_length, embed_size)
        seq_embed = self.word_embeddings(input_seqs)

        pos_embeds, neg_embeds = [], []
        for input_emoji in input_emojis:
            emoji_ids = torch.LongTensor(input_emoji).to(self.device)

            # shape: (num_emojis, emoji_embed_size)
            pos_emoji_embed = self.pos_emoji_embedding(emoji_ids)
            neg_emoji_embed = self.neg_emoji_embedding(emoji_ids)

            # shape: (1, emoji_embed_size)
            pos_emoji_embed = torch.mean(pos_emoji_embed, dim=0).unsqueeze(0)
            neg_emoji_embed = torch.mean(neg_emoji_embed, dim=0).unsqueeze(0)

            pos_embeds.append(pos_emoji_embed)
            neg_embeds.append(neg_emoji_embed)

        # shape: (batch_size, emoji_embed_size)
        pos_embeds = torch.cat(pos_embeds, 0)
        neg_embeds = torch.cat(neg_embeds, 0)

        # shape: (batch_size, 1, emoji_embed_size * 2)
        emoji_embed = torch.cat((pos_embeds, neg_embeds), 1).unsqueeze(1)

        vt = self.attention(emoji_embed, seq_embed)

        # shape: (batch, max_seq_length, word_embed_size + emoji_embed_size * 2)
        input_embed = torch.cat((seq_embed, vt), 2)

        # shape: (batch_size, max_seq_length, lstm_hidden_size * 2)
        lstm_output, _ = self.LSTM(input_embed)
        # shape: (batch_size, lstm_hidden_size * 2)
        lstm_output = lstm_output[:, -1, :]

        # shape: (batch_size, num_outputs)
        logits = self.sigmoid(self.fc(lstm_output))

        return logits

