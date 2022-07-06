"""
DeepMoji
:author: Qizhi Li
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class Config:
    def __init__(self, num_classes, embed_dropout_rate=0, final_dropout_rate=0, lstm_dropout=0):
        self.embedding_dim = 300
        self.lstm_hidden_size = 512
        self.attention_size = 4 * self.lstm_hidden_size + self.embedding_dim
        self.embed_dropout_rate = embed_dropout_rate
        self.final_dropout_rate = final_dropout_rate
        self.lstm_dropout = lstm_dropout
        self.num_classes = num_classes

        self.batch_size = 128
        self.max_seq_length = 64
        self.lr = 1e-5
        self.epoch = 50
        self.early_stop = 512


class Attention(nn.Module):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, attention_size, device, return_attention=False):
        """ Initialize the attention layer

        # Arguments:
            attention_size: Size of the attention vector.
            return_attention: If true, output will include the weight for each input token
                              used for the prediction

        """
        super(Attention, self).__init__()
        self.device = device
        self.return_attention = return_attention
        self.attention_size = attention_size
        self.attention_vector = Parameter(torch.FloatTensor(attention_size))
        self.attention_vector.data.normal_(std=0.05)  # Initialize attention vector

    def __repr__(self):
        s = '{name}({attention_size}, return attention={return_attention})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, input_lengths):
        """ Forward pass.

        # Arguments:
            inputs (Torch.Variable): Tensor of input sequences
            input_lengths (torch.LongTensor): Lengths of the sequences

        # Return:
            Tuple with (representations and attentions if self.return_attention else None).
        """
        # et = ht \times wa
        logits = inputs.matmul(self.attention_vector)
        # at = e^et / e^ei
        unnorm_ai = (logits - logits.max()).exp()

        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        max_len = unnorm_ai.size(1)
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        mask = Variable((idxes < input_lengths.unsqueeze(1)).float()).to(self.device)

        # apply mask and renormalize attention scores (weights)
        masked_weights = unnorm_ai * mask
        att_sums = masked_weights.sum(dim=1, keepdim=True)  # sums per sequence
        attentions = masked_weights.div(att_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(dim=1)

        return (representations, attentions if self.return_attention else None)


class DeepMoji(nn.Module):
    def __init__(self, embed, config, device):
        super().__init__()
        self.config = config
        self.device = device

        self.embeddings = nn.Embedding.from_pretrained(embed, freeze=False)
        self.lstm_0 = nn.LSTM(input_size=config.embedding_dim,
                              hidden_size=config.lstm_hidden_size,
                              num_layers=1,
                              dropout=config.lstm_dropout,
                              bidirectional=True,
                              batch_first=True)
        self.lstm_1 = nn.LSTM(input_size=config.lstm_hidden_size*2,
                              hidden_size=config.lstm_hidden_size,
                              num_layers=1,
                              dropout=config.lstm_dropout,
                              bidirectional=True,
                              batch_first=True)
        self.attention = Attention(config.attention_size, device)
        self.output_layer = nn.Linear(config.attention_size, config.num_classes)
        self.embed_dropout = nn.Dropout(config.embed_dropout_rate)
        self.final_dropout = nn.Dropout(config.final_dropout_rate)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

        nn.init.xavier_uniform(self.output_layer.weight.data)

    def forward(self, input_seqs):
        input_seqs = Variable(input_seqs)
        input_lengths = torch.LongTensor(
            [torch.max(input_seqs[i, :].data.nonzero()) + 1 for i in range(input_seqs.size()[0])])
        # input_lengths = input_lengths.to(self.device)
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        input_seqs = input_seqs[perm_idx][:, :input_lengths.max()]

        # Pack sequence and work on data tensor to reduce embeddings/dropout computations
        packed_input = pack_padded_sequence(input_seqs, input_lengths.cpu().numpy(), batch_first=True)

        ho = self.lstm_0.weight_hh_l0.data.new(2,
                                               input_seqs.size()[0],
                                               self.config.lstm_hidden_size).zero_().to(self.device)
        co = self.lstm_0.weight_hh_l0.data.new(2,
                                               input_seqs.size()[0],
                                               self.config.lstm_hidden_size).zero_().to(self.device)
        hidden = (Variable(ho, requires_grad=False), Variable(co, requires_grad=False))

        x = self.embeddings(packed_input.data)
        x = self.tanh(x)
        x = self.embed_dropout(x)

        packed_input = PackedSequence(x, packed_input.batch_sizes)

        lstm_0_output, _ = self.lstm_0(packed_input, hidden)
        lstm_1_output, _ = self.lstm_1(lstm_0_output, hidden)

        packed_input = PackedSequence(torch.cat((lstm_1_output.data,
                                                 lstm_0_output.data,
                                                 packed_input.data), dim=1),
                                      packed_input.batch_sizes)

        input_seqs, _ = pad_packed_sequence(packed_input, batch_first=True)

        x, att_weights = self.attention(input_seqs, input_lengths)

        x = self.final_dropout(x)
        outputs = self.softmax(self.output_layer(x))

        return outputs


