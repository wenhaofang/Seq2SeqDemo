# Transformer

import torch
import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super(MultiHeadAttentionLayer, self).__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask = None):
        '''
        Params:
            query : Torch LongTensor (batch_size, q_len, hid_dim)
            key   : Torch LongTensor (batch_size, k_ken, hid_dim)
            value : Torch LongTensor (batch_size, v_len, hid_dim)
            mask  : Torch LongTensor (batch_size, 1, 1      , src_len) if in encoder
                    Torch LongTensor (batch_size, 1, trg_len, trg_len) if in decoder
        Return:
            x     : Torch LongTensor (batch_size, seq_len, hid_dim)
        '''
        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim = -1)

        x = torch.matmul(self.dropout(attention), V)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)

        x = self.fc_o(x)
        return x

class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, hid_dim, mid_dim, dropout):
        super(PositionWiseFeedForwardLayer, self).__init__()

        self.fc_1 = nn.Linear(hid_dim, mid_dim)
        self.fc_2 = nn.Linear(mid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        Params:
            x : Torch LongTensor (batch_size, seq_len, hid_dim)
        Return:
            x : Torch LongTensor (batch_size, seq_len, hid_dim)
        '''
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, mid_dim, dropout, device):
        super(EncoderLayer, self).__init__()

        self.multi_head_self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.position_wise_feedforward = PositionWiseFeedForwardLayer(hid_dim, mid_dim, dropout)

        self.self_attention_layer_norm = nn.LayerNorm(hid_dim)
        self.feed_forward_layer_norm = nn.LayerNorm(hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        '''
        Params:
            src      : Torch LongTensor (batch_size, src_len, hid_dim)
            src_mask : Torch LongTensor (batch_size, 1, 1, src_len)
        Return:
            src      : Torch LongTensor (batch_size, src_len, hid_dim)
        '''
        _src = self.multi_head_self_attention(src, src, src, src_mask)
        src  = self.self_attention_layer_norm(src + self.dropout(_src))

        _src = self.position_wise_feedforward(src)
        src  = self.feed_forward_layer_norm  (src + self.dropout(_src))

        return src

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, mid_dim, dropout, device):
        super(DecoderLayer, self).__init__()

        self.multi_head_self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.multi_head_enco_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.position_wise_feedforward = PositionWiseFeedForwardLayer(hid_dim, mid_dim, dropout)

        self.self_attention_layer_norm = nn.LayerNorm(hid_dim)
        self.enco_attention_layer_norm = nn.LayerNorm(hid_dim)
        self.feed_forward_layer_norm = nn.LayerNorm(hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask, src_mask):
        '''
        Params:
            trg      : Torch LongTensor (batch_size, trg_len, hid_dim)
            src      : Torch LongTensor (batch_size, src_len, hid_dim)
            trg_mask : Torch LongTensor (batch_size, 1, trg_len, trg_len)
            src_mask : Torch LongTensor (batch_size, 1, 1, src_len)
        Return:
            trg      : Torch LongTensor (batch_size, trg_len, hid_dim)
        '''
        _trg = self.multi_head_self_attention(trg, trg, trg, trg_mask)
        trg  = self.self_attention_layer_norm(trg + self.dropout(_trg))

        _trg = self.multi_head_enco_attention(trg, src, src, src_mask)
        trg  = self.enco_attention_layer_norm(trg + self.dropout(_trg))

        _trg = self.position_wise_feedforward(trg)
        trg  = self.feed_forward_layer_norm  (trg + self.dropout(_trg))

        return trg

class Encoder(nn.Module):
    def __init__(self, vocab_size, hid_dim, n_heads, mid_dim, n_layers, dropout, device, max_length = 100):
        super(Encoder, self).__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(vocab_size, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([
            EncoderLayer(
                hid_dim, n_heads, mid_dim, dropout, device
            ) for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        '''
        Params:
            src      : Torch LongTensor (batch_size, src_len)
            src_mask : Torch LongTensor (batch_size, 1, 1, src_len)
        Return:
            embedded : Torch LongTensor (batch_size, src_len, hid_dim)
        '''
        batch_size, src_len = src.shape[0], src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)
        embedded = self.dropout(tok_embedded * self.scale + pos_embedded)

        for layer in self.layers:
            embedded = layer(embedded, src_mask)

        return embedded

class Decoder(nn.Module):
    def __init__(self, vocab_size, hid_dim, n_heads, mid_dim, n_layers, dropout, device, max_length = 100):
        super(Decoder, self).__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(vocab_size, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([
            DecoderLayer(
                hid_dim, n_heads, mid_dim, dropout, device
            ) for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        self.fc_out = nn.Linear(hid_dim, vocab_size)

    def forward(self, trg, src_emb, trg_mask, src_mask):
        '''
        Params:
            trg      : Torch LongTensor (batch_size, trg_len)
            src_emb  : Torch LongTensor (batch_size, src_len, hid_dim)
            trg_mask : Torch LongTensor (batch_size, 1, trg_len, trg_len)
            src_mask : Torch LongTensor (batch_size, 1, 1, src_len)
        Return:
            output   : Torch LongTensor (batch_size, trg_len, trg_vocab_size)
        '''
        batch_size, trg_len = trg.shape[0], trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        tok_embedded = self.tok_embedding(trg)
        pos_embedded = self.pos_embedding(pos)
        embedded = self.dropout(tok_embedded * self.scale + pos_embedded)

        for layer in self.layers:
            embedded = layer(embedded, src_emb, trg_mask, src_mask)

        output = self.fc_out(embedded)

        return output

class Seq2Seq(nn.Module):
    def __init__(self,
        src_vocab_size, enc_hid_dim, enc_attention_heads, enc_mid_dim, enc_transformer_layers, enc_dropout, src_pad_idx, device,
        trg_vocab_size, dec_hid_dim, dec_attention_heads, dec_mid_dim, dec_transformer_layers, dec_dropout, trg_pad_idx
    ):
        super(Seq2Seq, self).__init__()

        assert enc_hid_dim == dec_hid_dim
        assert enc_hid_dim % enc_attention_heads == 0
        assert dec_hid_dim % dec_attention_heads == 0

        self.device = device

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.encoder = Encoder(src_vocab_size, enc_hid_dim, enc_attention_heads, enc_mid_dim, enc_transformer_layers, enc_dropout, device)
        self.decoder = Decoder(trg_vocab_size, dec_hid_dim, dec_attention_heads, dec_mid_dim, dec_transformer_layers, dec_dropout, device)

    def make_src_mask(self, src):
        '''
        Params:
            src: Torch LongTensor (batch_size, src_len)
        Return:
            src_pad_mask: Torch LongTensor (batch_size, 1, 1, src_len)
        '''
        src_pad_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_pad_mask

    def make_trg_mask(self, trg):
        '''
        Params:
            trg: Torch LongTensor (batch_size, trg_len)
        Return:
            trg_mask: Torch LongTensor (batch_size, 1, trg_len, trg_len)
        '''
        trg_len = trg.shape[1]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()

        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        '''
        Params:
            src: Torch LongTensor (batch_size, src_len)
            trg: Torch LongTensor (batch_size, trg_len)
        Return:
            output: Torch LongTensor (batch_size, trg_len, trg_vocab_size)
        '''
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        src_emb = self.encoder(src, src_mask)
        output  = self.decoder(trg, src_emb, trg_mask, src_mask)

        return output

def get_module(option, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device):
    seq2seq = Seq2Seq(
        src_vocab_size, option.enc_hid_dim, option.enc_attention_heads, option.enc_mid_dim, option.enc_transformer_layers, option.enc_dropout, src_pad_idx, device,
        trg_vocab_size, option.dec_hid_dim, option.dec_attention_heads, option.dec_mid_dim, option.dec_transformer_layers, option.dec_dropout, trg_pad_idx
    )

    return seq2seq

if __name__ == '__main__':
    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    src_vocab_size = 13000
    trg_vocab_size = 17000
    src_padded_idx = 0
    trg_padded_idx = 0

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    module = get_module(option, src_vocab_size, trg_vocab_size, src_padded_idx, trg_padded_idx, device)
    module = module.to (device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    para_num = count_parameters(module)
    print(f'The module has {para_num} trainable parameters')

    src = torch.zeros((16, 18)).long().to(device) # (batch_size, src_seq_len)
    trg = torch.zeros((16, 20)).long().to(device) # (batch_size, src_seq_len)

    pred = module(src, trg) # (batch_size, trg_seq_len, trg_vocab_size)
