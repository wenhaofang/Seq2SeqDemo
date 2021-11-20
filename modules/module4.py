# Pack & Pad, Attention Mask

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class Encoder(nn.Module):
    def __init__( self,
        rnn_type, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, n_directions, dropout
    ):
        super(Encoder, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.rnn_type = rnn_type
        if rnn_type == 'rnn':
            self.rnn = nn.RNN (emb_dim, enc_hid_dim, num_layers = n_layers, bidirectional = True if n_directions == 2 else False)
        if rnn_type == 'gru':
            self.rnn = nn.GRU (emb_dim, enc_hid_dim, num_layers = n_layers, bidirectional = True if n_directions == 2 else False)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim, enc_hid_dim, num_layers = n_layers, bidirectional = True if n_directions == 2 else False)

        self.linear = nn.Linear(enc_hid_dim * n_layers * n_directions, dec_hid_dim)

    def forward(self, source, source_length): # diff: source_length
        '''
        Params:
            source: Torch LongTensor (src_seq_len, batch_size)
            source_length: Torch LongTensor (batch_size)
        Return:
            outputs: Torch LongTensor (src_seq_len, batch_size, n_directions * enc_hid_dim)
            hidden : Torch LongTensor (batch_size, dec_hid_dim) if rnn_type == 'rnn' or rnn_type == 'gru'
                    (
                        Torch LongTensor (batch_size, dec_hid_dim),
                        Torch LongTensor (batch_size, dec_hid_dim)
                    ) if rnn_type == 'lstm'
        '''
        embedded = self.dropout(self.embedding(source))
        embedded_packed = pack_padded_sequence(embedded, source_length.to('cpu')) # diff: pack_padded_sequence
        outputs_packed, hidden = self.rnn(embedded_packed)                        # diff
        outputs, length_unpacked = pad_packed_sequence(outputs_packed)            # diff: pad_packed_sequence
        if self.rnn_type == 'rnn' or self.rnn_type == 'gru':
            hidden = torch.tanh(self.linear(torch.cat([hidden[i] for i in range(hidden.shape[0])], dim = -1)))
        if self.rnn_type == 'lstm':
            hidden = (
                torch.tanh(self.linear(torch.cat([hidden[0][i] for i in range(hidden[0].shape[0])], dim = -1))),
                torch.tanh(self.linear(torch.cat([hidden[1][i] for i in range(hidden[1].shape[0])], dim = -1))) # make sense?
            )
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, enc_n_directions, dec_hid_dim):
        super(Attention, self).__init__()
        self.a = nn.Linear(enc_hid_dim * enc_n_directions + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim , 1, bias = False)

    def forward(self, hidden, encoder_outputs, masked): # diff: masked
        '''
        Params:
            hidden : Torch LongTensor (batch_size, dec_hid_dim)
            outputs: Torch LongTensor (src_seq_len, batch_size, enc_n_directions * enc_hid_dim)
            masked : Torch LongTensor (batch_size, src_seq_len)
        Return:
            attention: Torch LongTensor (batch_size, src_seq_len)
        '''
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        concated = torch.cat((hidden, encoder_outputs), dim = 2)

        energy    = torch.tanh(self.a(concated))
        attention = F.softmax (self.v(energy).squeeze(2).masked_fill(masked == 0, -1e10), dim = 1)  # diff: masked_fill

        return attention

class Decoder(nn.Module):
    def __init__( self,
        rnn_type, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, enc_n_directions, dec_n_directions, dropout
    ):
        super(Decoder, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.attention = Attention(enc_hid_dim, enc_n_directions, dec_hid_dim)

        self.rnn_type = rnn_type
        if rnn_type == 'rnn':
            self.rnn = nn.RNN (enc_hid_dim * enc_n_directions + emb_dim , dec_hid_dim, num_layers = n_layers, bidirectional = True if dec_n_directions == 2 else False)
        if rnn_type == 'gru':
            self.rnn = nn.GRU (enc_hid_dim * enc_n_directions + emb_dim , dec_hid_dim, num_layers = n_layers, bidirectional = True if dec_n_directions == 2 else False)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(enc_hid_dim * enc_n_directions + emb_dim , dec_hid_dim, num_layers = n_layers, bidirectional = True if dec_n_directions == 2 else False)

        self.linear =nn.Linear(enc_hid_dim * enc_n_directions + emb_dim + dec_hid_dim, vocab_size)

    def forward(self, source, hidden, encoder_outputs, masked): # diff: masked
        '''
        Params:
            source : Torch LongTensor (batch_size)
            masked : Torch LongTensor (batch_size, src_seq_len)
            outputs: Torch LongTensor (src_seq_len, batch_size, enc_n_directions * enc_hid_dim)
            hidden : Torch LongTensor (batch_size, dec_hid_dim) if rnn_type == 'rnn' or rnn_type == 'gru'
                    (
                        Torch LongTensor (batch_size, dec_hid_dim),
                        Torch LongTensor (batch_size, dec_hid_dim)
                    ) if rnn_type == 'lstm'
        Return:
            pred..: Torch LongTensor (batch_size, trg_vocab_size)
            hidden: Torch LongTensor (batch_size, dec_hid_dim) if rnn_type == 'rnn' or rnn_type == 'gru'
                    (
                        Torch LongTensor (batch_size, dec_hid_dim),
                        Torch LongTensor (batch_size, dec_hid_dim)
                    ) if rnn_type == 'lstm'
        '''
        embedded = self.dropout(self.embedding(source.unsqueeze(0)))

        if self.rnn_type == 'rnn' or self.rnn_type == 'gru':
            score = self.attention(hidden, encoder_outputs, masked).unsqueeze(1) # diff: masked
        if self.rnn_type == 'lstm':
            score = self.attention(hidden[0], encoder_outputs, masked).unsqueeze(1) # diff: masked

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
        src_vocab_size, enc_emb_dim, enc_hid_dim, enc_n_layers, enc_n_directions, enc_dropout, src_padded_idx, # diff: src_padded_idx
        trg_vocab_size, dec_emb_dim, dec_hid_dim, dec_n_layers, dec_n_directions, dec_dropout
    ):
        super(Seq2Seq, self).__init__()
        self.trg_vocab_size = trg_vocab_size # diff: src_padded_idx
        self.src_padded_idx = src_padded_idx

        assert rnn_type in ['rnn', 'gru', 'lstm']
        assert dec_n_layers == 1
        assert dec_n_directions == 1

        self.encoder = Encoder(rnn_type, src_vocab_size, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_n_layers, enc_n_directions, enc_dropout)
        self.decoder = Decoder(rnn_type, trg_vocab_size, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_n_layers, enc_n_directions, dec_n_directions, dec_dropout)

    def create_mask(self, source): # create_mask
        return (source != self.src_padded_idx).permute(1, 0)

    def forward(self, source, source_length, target, teacher_forcing_ratio = 0.5): # diff: source_length
        trg_len, batch_size = target.shape[0], target.shape[1]
        outputs = torch.zeros((trg_len, batch_size, self.trg_vocab_size), device = source.device)

        encoder_outputs, hidden = self.encoder(source, source_length) # diff: source_length
        inputs = target[0, :]
        masked = self.create_mask(source) # diff: create_mask

        for t in range(1, trg_len):
            output, hidden = self.decoder(inputs, hidden, encoder_outputs, masked) # diff: masked
            outputs [t] = output
            if random.random() < teacher_forcing_ratio:
                inputs = target[t]
            else:
                inputs = output.argmax(1)
        return outputs

def get_module(option, src_vocab_size, trg_vocab_size, src_padded_idx): # diff: src_padded_idx
    seq2seq = Seq2Seq(option.rnn_type,
        src_vocab_size, option.enc_emb_dim, option.enc_hid_dim, option.enc_n_layers, option.enc_n_directions, option.enc_dropout, src_padded_idx,  # diff: src_padded_idx
        trg_vocab_size, option.dec_emb_dim, option.dec_hid_dim, option.dec_n_layers, option.dec_n_directions, option.dec_dropout
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
    src_padded_idx = 0

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    module = get_module(option, src_vocab_size, trg_vocab_size, src_padded_idx)
    module = module.to (device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    para_num = count_parameters(module)
    print(f'The module has {para_num} trainable parameters')

    # src_len must in descending order, and the max value should be equal to src.shape[0]
    src_len = torch.tensor(range(18, 2, -1)).long().to(device)

    src = torch.zeros((18, 16)).long().to(device)
    trg = torch.zeros((20, 16)).long().to(device)

    pred = module(src, src_len, trg) # (trg_len, batch_size, trg_vocab_size)
