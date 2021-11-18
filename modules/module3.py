# Alleviate Information Compression of Encoder

# In the previous model, the context vector still needs to contain all of the information about the source sentence.

# This section will utilizes attention mechanism to allow the decoder to look at the entire source sentence at each decoding step.

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__( self,
        rnn_type, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, n_directions, dropout
    ):
        super(Encoder, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        assert rnn_type in ['rnn', 'gru', 'lstm']
        self.rnn_type = rnn_type

        if rnn_type == 'rnn':
            self.rnn = nn.RNN (emb_dim, enc_hid_dim, num_layers = n_layers, bidirectional = True if n_directions == 2 else False)
        if rnn_type == 'gru':
            self.rnn = nn.GRU (emb_dim, enc_hid_dim, num_layers = n_layers, bidirectional = True if n_directions == 2 else False)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim, enc_hid_dim, num_layers = n_layers, bidirectional = True if n_directions == 2 else False)

        self.linear = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, source):
        embedded = self.dropout(self.embedding(source))
        outputs, hidden = self.rnn(embedded)

        if self.rnn_type == 'rnn' or self.rnn_type == 'gru':
            hidden = torch.tanh(self.linear(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        if self.rnn_type == 'lstm':
            hidden = (
                torch.tanh(self.linear(torch.cat((hidden[0][-2,:,:], hidden[0][-1,:,:]), dim = 1))),
                torch.tanh(self.linear(torch.cat((hidden[0][-2,:,:], hidden[0][-1,:,:]), dim = 1))) # make sense?
            )

        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        self.a = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim , 1 , bias = False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        concated = torch.cat((hidden, encoder_outputs), dim = 2)

        energy    = torch.tanh(self.a(concated))
        attention = F.softmax (self.v(energy).squeeze(2), dim = 1)

        return attention

class Decoder(nn.Module):
    def __init__( self,
        rnn_type, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, n_directions, dropout
    ):
        super(Decoder, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.attention = Attention(enc_hid_dim, dec_hid_dim)

        assert rnn_type in ['rnn', 'gru', 'lstm']
        self.rnn_type = rnn_type

        if rnn_type == 'rnn':
            self.rnn = nn.RNN (enc_hid_dim * 2 + emb_dim , dec_hid_dim, num_layers = n_layers, bidirectional = True if n_directions == 2 else False)
        if rnn_type == 'gru':
            self.rnn = nn.GRU (enc_hid_dim * 2 + emb_dim , dec_hid_dim, num_layers = n_layers, bidirectional = True if n_directions == 2 else False)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(enc_hid_dim * 2 + emb_dim , dec_hid_dim, num_layers = n_layers, bidirectional = True if n_directions == 2 else False)

        self.linear =nn.Linear(enc_hid_dim * 2 + emb_dim + dec_hid_dim, vocab_size)

    def forward(self, source, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(source.unsqueeze(0)))

        if self.rnn_type == 'rnn' or self.rnn_type == 'gru':
            score = self.attention(hidden, encoder_outputs).unsqueeze(1)
        if self.rnn_type == 'lstm':
            score = self.attention(hidden[0], encoder_outputs).unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted = torch.bmm(score, encoder_outputs).permute(1, 0, 2)

        concated = torch.cat((embedded, weighted), dim = 2)

        if self.rnn_type == 'rnn' or self.rnn_type == 'gru':
            hidden = hidden.unsqueeze(0)
        if self.rnn_type == 'lstm':
            hidden = (
                hidden[0].unsqueeze(0), hidden[1].unsqueeze(0)
            )

        output, hidden = self.rnn(concated, hidden)

        if self.rnn_type == 'rnn' or self.rnn_type == 'gru':
            hidden = hidden.squeeze(0)
        if self.rnn_type == 'lstm':
            hidden = (
                hidden[0].squeeze(0), hidden[1].squeeze(0)
            )

        prediction = self.linear(torch.cat((output.squeeze(0), embedded.squeeze(0), weighted.squeeze(0)), dim = 1))

        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__( self, rnn_type,
        src_vocab_size, enc_emb_dim, enc_hid_dim, enc_n_layers, enc_n_directions, enc_dropout,
        trg_vocab_size, dec_emb_dim, dec_hid_dim, dec_n_layers, dec_n_directions, dec_dropout
    ):
        super(Seq2Seq, self).__init__()
        self.trg_vocab_size = trg_vocab_size

        self.encoder = Encoder(rnn_type, src_vocab_size, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_n_layers, enc_n_directions, enc_dropout)
        self.decoder = Decoder(rnn_type, trg_vocab_size, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_n_layers, dec_n_directions, dec_dropout)

    def forward(self, source, target, teacher_forcing_ratio = 0.5):
        trg_len, batch_size = target.shape[0], target.shape[1]
        outputs = torch.zeros((trg_len, batch_size, self.trg_vocab_size), device = source.device)

        encoder_outputs, hidden = self.encoder(source)
        inputs = target[0, :]

        for t in range(1, trg_len):
            output, hidden = self.decoder(inputs, hidden, encoder_outputs)
            outputs [t] = output
            if random.random() < teacher_forcing_ratio:
                inputs = target[t]
            else:
                inputs = output.argmax(1)
        return outputs

def get_module(option, src_vocab_size, trg_vocab_size):
    seq2seq = Seq2Seq(option.rnn_type,
        src_vocab_size, option.enc_emb_dim, option.enc_hid_dim, option.enc_n_layers_group1, option.enc_n_directions_group2, option.enc_dropout,
        trg_vocab_size, option.dec_emb_dim, option.dec_hid_dim, option.dec_n_layers_group1, option.dec_n_directions_group1, option.dec_dropout
    )

    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean = 0, std = 0.01)
            else:
                nn.init.constant_(param.data, 0)

    seq2seq.apply(init_weights)

    return seq2seq

if __name__ == '__main__':
    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    src_vocab_size = 13000
    trg_vocab_size = 17000

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    module = get_module(option, src_vocab_size, trg_vocab_size)
    module = module.to (device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    para_num = count_parameters(module)
    print(f'The module has {para_num} trainable parameters')

    src = torch.zeros((18, 16)).long().to(device)
    trg = torch.zeros((20, 16)).long().to(device)
    pred = module(src, trg) # (trg_len, batch_size, trg_vocab_size)
