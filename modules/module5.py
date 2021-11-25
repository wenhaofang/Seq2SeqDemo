# Convolutional Seq2Seq

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, dropout, kernel_size, filter_layers, device, max_length = 100):
        super(Encoder, self).__init__()

        self.device = device

        self.scale = torch.torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels  = hid_dim,
                out_channels = hid_dim * 2,
                kernel_size  = kernel_size,
                padding = (kernel_size - 1) // 2
            ) for _ in range(filter_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        '''
        Params:
            src: Torch LongTensor (batch_size, src_seq_len)
        Return:
            conv_output: Torch LongTensor (batch_size, src_seq_len, enc_emb_dim)
            combined   : Torch LongTensor (batch_size, src_seq_len, enc_emb_dim)
        '''
        batch_size, src_len = src.shape[0], src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)
        embedded = self.dropout(tok_embedded + pos_embedded)

        conv_input = self.emb2hid(embedded).permute(0, 2, 1)

        for conv in self.convs:
            conv_input = self.dropout(conv_input)

            conved = conv (conv_input)
            conved = F.glu(conved, dim = 1)

            conv_input = (conved + conv_input) * self.scale

        conv_output = self.hid2emb(conv_input.permute(0, 2, 1))

        combined = (conv_output + embedded) * self.scale

        return conv_output, combined

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, dropout, kernel_size, filter_layers, device, trg_pad_idx, max_length = 100):
        super(Decoder, self).__init__()

        self.device = device

        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx

        self.scale = torch.torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        
        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)

        self.fc_out = nn.Linear(emb_dim, vocab_size)

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels  = hid_dim,
                out_channels = hid_dim * 2,
                kernel_size  = kernel_size
            ) for _ in range(filter_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def create_padding(self , conv_input):
        batch_size, hid_dim = conv_input.shape[0], conv_input.shape[1]
        padding = torch.zeros(batch_size, hid_dim, self.kernel_size-1).fill_(self.trg_pad_idx).to(self.device)
        return padding

    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))
        combined = (conved_emb + embedded) * self.scale

        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))
        attention = F.softmax(energy, dim = 2)

        attended_encoding = torch.matmul(attention, encoder_combined)
        attended_encoding = self.attn_emb2hid(attended_encoding)

        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale

        return attended_combined

    def forward(self, trg, encoder_conved, encoder_combined):
        '''
        Params:
            trg: Torch LongTensor (batch_size, src_seq_len)
            encoder_conved  : Torch LongTensor (batch_size, src_seq_len, enc_emb_dim)
            encoder_combined: Torch LongTensor (batch_size, src_seq_len, enc_emb_dim)
        Return:
            pred_output: Torch LongTensor (batch_size, trg_seq_len, trg_vocab_size)
        '''
        batch_size, trg_len = trg.shape[0], trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        tok_embedded = self.tok_embedding(trg)
        pos_embedded = self.pos_embedding(pos)
        embedded = self.dropout(tok_embedded + pos_embedded)

        conv_input = self.emb2hid(embedded).permute(0, 2, 1)

        for conv in self.convs:
            conv_input = self.dropout(conv_input)

            padding = self.create_padding(conv_input)
            padded_conv_input = torch.cat((padding, conv_input), dim = 2)

            conved = conv (padded_conv_input)
            conved = F.glu(conved, dim = 1)

            conved = self.calculate_attention(embedded, conved, encoder_conved, encoder_combined)

            conv_input = (conved + conv_input) * self.scale

        conv_output = self.hid2emb(conv_input.permute(0, 2, 1))

        pred_output = self.fc_out(self.dropout(conv_output))

        return pred_output

class Seq2Seq(nn.Module):
    def __init__(self,
        src_vocab_size, enc_emb_dim, enc_hid_dim, enc_dropout, enc_kernel_size, enc_filter_layers, device,
        trg_vocab_size, dec_emb_dim, dec_hid_dim, dec_dropout, dec_kernel_size, dec_filter_layers, trg_pad_idx
    ):
        super(Seq2Seq, self).__init__()

        assert enc_emb_dim == dec_emb_dim
        assert enc_kernel_size % 2 == 1

        self.encoder = Encoder(src_vocab_size, enc_emb_dim, enc_hid_dim, enc_dropout, enc_kernel_size, enc_filter_layers, device)
        self.decoder = Decoder(trg_vocab_size, dec_emb_dim, dec_hid_dim, dec_dropout, dec_kernel_size, dec_filter_layers, device, trg_pad_idx)

    def forward(self, src, trg):
         encoder_conved, encoder_combined = self.encoder(src)
         output = self.decoder(trg, encoder_conved, encoder_combined)
         return output

def get_module(option, src_vocab_size, trg_vocab_size, trg_pad_idx, device):
    seq2seq = Seq2Seq(
        src_vocab_size, option.enc_emb_dim, option.enc_hid_dim, option.enc_dropout, option.enc_kernel_size, option.enc_filter_layers, device,
        trg_vocab_size, option.dec_emb_dim, option.dec_hid_dim, option.dec_dropout, option.dec_kernel_size, option.dec_filter_layers, trg_pad_idx
    )

    return seq2seq

if __name__ == '__main__':
    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    src_vocab_size = 13000
    trg_vocab_size = 17000
    trg_padded_idx = 0

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    module = get_module(option, src_vocab_size, trg_vocab_size, trg_padded_idx, device)
    module = module.to (device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    para_num = count_parameters(module)
    print(f'The module has {para_num} trainable parameters')

    src = torch.zeros((16, 18)).long().to(device) # (batch_size, src_seq_len)
    trg = torch.zeros((16, 20)).long().to(device) # (batch_size, trg_seq_len)

    pred = module(src, trg) # (batch_size, trg_seq_len, trg_vocab_size)
