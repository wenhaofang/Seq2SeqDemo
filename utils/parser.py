import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # For Basic
    parser.add_argument('--name', default = 'main', help = '')

    # For Loader
    parser.add_argument('--sources_path', default = 'datasources', help = '')
    parser.add_argument('--targets_path', default = 'datatargets', help = '')

    parser.add_argument('--train_file', default = 'train_data.tsv', help = '')
    parser.add_argument('--valid_file', default = 'valid_data.tsv', help = '')
    parser.add_argument('--test_file' , default = 'test_data.tsv' , help = '')

    parser.add_argument('--min_freq', type = int, default = 7, help = '')
    parser.add_argument('--max_numb', type = int, default = 30000, help = '')
    parser.add_argument('--max_seq_len', type = int, default = 16, help = '')

    # For Module
    parser.add_argument('--rnn_type', choices = ['rnn', 'gru','lstm'], default = 'rnn', help = '')

    parser.add_argument('--enc_emb_dim', type = int, default = 256, help = '')
    parser.add_argument('--dec_emb_dim', type = int, default = 256, help = '')
    parser.add_argument('--enc_hid_dim', type = int, default = 512, help = '')
    parser.add_argument('--dec_hid_dim', type = int, default = 512, help = '')
    parser.add_argument('--enc_dropout', type = float, default = 0.5, help = '')
    parser.add_argument('--dec_dropout', type = float, default = 0.5, help = '')

    parser.add_argument('--enc_n_layers_group1', type = int, default = 1, help = '')
    parser.add_argument('--dec_n_layers_group1', type = int, default = 1, help = '')
    parser.add_argument('--enc_n_layers_group2', type = int, default = 2, help = '')
    parser.add_argument('--dec_n_layers_group2', type = int, default = 2, help = '')

    parser.add_argument('--enc_n_directions_group1', type = int, default = 1, help = '')
    parser.add_argument('--dec_n_directions_group1', type = int, default = 1, help = '')
    parser.add_argument('--enc_n_directions_group2', type = int, default = 2, help = '')
    parser.add_argument('--dec_n_directions_group2', type = int, default = 2, help = '')

    # For Train
    parser.add_argument('--module', type = int, choices = range(1, 7), default = 1, help = '')

    parser.add_argument('--batch_size', type = int, default = 64, help = '')
    parser.add_argument('--num_epochs', type = int, default = 10, help = '')

    parser.add_argument('--grad_clip', type = float, default = 1, help = '')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    option = parser.parse_args()
