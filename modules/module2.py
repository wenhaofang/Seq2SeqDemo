# Alleviate Information Compression of Decoder

# In the previous model, decoder try to cram lots of information into hidden state.
# While decoding, the hidden state will need to contain information about the whole source sequence, as well as the tokens have been decoded so far.

# So what should we do?
# 1. Instead of the rnn in the decoder taking just the embedded target token and the previous hidden state as inputs, it also takes the context vector.
# 2. Instead of the linear layer in the decoder taking just the hidden state at that time-step as inputs, it also takes the embedding of current token and the context vector.

import random

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__( self,
        rnn_type, vocab_size, emb_dim, hid_dim, n_layers, n_directions, dropout
    ):
        super(Encoder, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        if rnn_type == 'rnn':
            self.rnn = nn.RNN (emb_dim, hid_dim, num_layers = n_layers, bidirectional = True if n_directions == 2 else False)
        if rnn_type == 'gru':
            self.rnn = nn.GRU (emb_dim, hid_dim, num_layers = n_layers, bidirectional = True if n_directions == 2 else False)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers = n_layers, bidirectional = True if n_directions == 2 else False)

    def forward(self, source):
        '''
        Params:
            source: Torch LongTensor (src_seq_len, batch_size)
        Return:
            hidden: Torch LongTensor (n_directions * n_layers, batch_size, enc_hid_dim) if rnn_type == 'rnn' or rnn_type == 'gru'
                    (
                        Torch LongTensor (n_directions * n_layers, batch_size, enc_hid_dim),
                        Torch LongTensor (n_directions * n_layers, batch_size, enc_hid_dim)
                    ) if rnn_type == 'lstm'
        '''
        embedded = self.dropout(self.embedding(source))
        _,hidden = self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__( self,
        rnn_type, vocab_size, emb_dim, hid_dim, n_layers, n_directions, dropout
    ):
        super(Decoder, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.rnn_type = rnn_type
        if rnn_type == 'rnn':
            self.rnn = nn.RNN (emb_dim + hid_dim, hid_dim, num_layers = n_layers, bidirectional = True if n_directions == 2 else False)
        if rnn_type == 'gru':
            self.rnn = nn.GRU (emb_dim + hid_dim, hid_dim, num_layers = n_layers, bidirectional = True if n_directions == 2 else False)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, num_layers = n_layers, bidirectional = True if n_directions == 2 else False)

        self.linear =nn.Linear(emb_dim + hid_dim * 2, vocab_size)

    def forward(self, source, hidden, context):
        '''
        Params:
            source : Torch LongTensor (batch_size)
            hidden : Torch LongTensor (n_directions * n_layers, batch_size, dec_hid_dim) if rnn_type == 'rnn' or rnn_type == 'gru'
                    (
                        Torch LongTensor (n_directions * n_layers, batch_size, dec_hid_dim),
                        Torch LongTensor (n_directions * n_layers, batch_size, dec_hid_dim)
                    ) if rnn_type == 'lstm'
            context: Torch LongTensor (n_directions * n_layers, batch_size, enc_hid_dim) if rnn_type == 'rnn' or rnn_type == 'gru'
                    (
                        Torch LongTensor (n_directions * n_layers, batch_size, enc_hid_dim),
                        Torch LongTensor (n_directions * n_layers, batch_size, enc_hid_dim)
                    ) if rnn_type == 'lstm'
        Return:
            pred..: Torch LongTensor (batch_size, trg_vocab_size)
            hidden: Torch LongTensor (n_directions * n_layers, batch_size, dec_hid_dim) if rnn_type == 'rnn' or rnn_type == 'gru'
                    (
                        Torch LongTensor (n_directions * n_layers, batch_size, dec_hid_dim),
                        Torch LongTensor (n_directions * n_layers, batch_size, dec_hid_dim)
                    ) if rnn_type == 'lstm'
        '''
        embedded = self.dropout(self.embedding(source.unsqueeze(0)))
        if self.rnn_type == 'rnn':
            concated = torch.cat((embedded, context[-1].unsqueeze(0)), dim = 2)
        if self.rnn_type == 'gru':
            concated = torch.cat((embedded, context[-1].unsqueeze(0)), dim = 2)
        if self.rnn_type == 'lstm':
            concated = torch.cat((embedded, context[0][-1].unsqueeze(0)), dim = 2)
        _, hidden = self.rnn(concated, hidden)
        if self.rnn_type == 'rnn':
            output = torch.cat((embedded.squeeze(0), hidden[-1], context[-1]), dim = 1)
        if self.rnn_type == 'gru':
            output = torch.cat((embedded.squeeze(0), hidden[-1], context[-1]), dim = 1)
        if self.rnn_type == 'lstm':
            output = torch.cat((embedded.squeeze(0), hidden[0][-1], context[0][-1]), dim = 1)
        prediction = self.linear(output)
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__( self, rnn_type,
        src_vocab_size, enc_emb_dim, enc_hid_dim, enc_n_layers, enc_n_directions, enc_dropout,
        trg_vocab_size, dec_emb_dim, dec_hid_dim, dec_n_layers, dec_n_directions, dec_dropout
    ):
        super(Seq2Seq, self).__init__()
        self.trg_vocab_size = trg_vocab_size

        assert rnn_type in ['rnn', 'gru', 'lstm']
        assert enc_hid_dim == dec_hid_dim
        assert enc_n_layers == dec_n_layers
        assert enc_n_directions == 1
        assert dec_n_directions == 1

        self.encoder = Encoder(rnn_type, src_vocab_size, enc_emb_dim, enc_hid_dim, enc_n_layers, enc_n_directions, enc_dropout)
        self.decoder = Decoder(rnn_type, trg_vocab_size, dec_emb_dim, dec_hid_dim, dec_n_layers, dec_n_directions, dec_dropout)

    def forward(self, source, target, teacher_forcing_ratio = 0.5):
        '''
        Params:
            source: Torch LongTensor (src_seq_len, batch_size)
            target: Torch LongTensor (trg_seq_len, batch_size)
            teacher_forcing_ratio: float between 0 and 1
        Return:
            outputs: Torch LongTensor (trg_seq_len, batch_size, trg_vocab_size)
        '''
        trg_len, batch_size = target.shape[0], target.shape[1]
        outputs = torch.zeros((trg_len, batch_size, self.trg_vocab_size), device = source.device)

        hidden = context = self.encoder(source)
        inputs = target[0, :]

        for t in range(1, trg_len):
            output, hidden = self.decoder(inputs, hidden, context)
            outputs [t] = output
            if random.random() < teacher_forcing_ratio:
                inputs = target[t]
            else:
                inputs = output.argmax(1)
        return outputs

def get_module(option, src_vocab_size, trg_vocab_size):
    seq2seq = Seq2Seq(option.rnn_type,
        src_vocab_size, option.enc_emb_dim, option.enc_hid_dim, option.enc_n_layers, option.enc_n_directions, option.enc_dropout,
        trg_vocab_size, option.dec_emb_dim, option.dec_hid_dim, option.dec_n_layers, option.dec_n_directions, option.dec_dropout
    )

    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.normal_(param.data, mean = 0, std = 0.01)

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
