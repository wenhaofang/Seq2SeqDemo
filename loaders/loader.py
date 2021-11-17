import os
import spacy
import torch

SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'

spacy_de = spacy.load('de_core_news_sm') # python -m spacy download de_core_news_sm
spacy_en = spacy.load('en_core_web_sm' ) # python -m spacy download en_core_web_sm

def read_data(data_path):
    data = []
    with open(data_path, 'r', encoding = 'utf-8') as data_file:
        for line in data_file:
            src_sent, trg_sent = line.split('\t')
            src_sent = src_sent.strip()
            trg_sent = trg_sent.strip()
            data.append((src_sent, trg_sent))
    return data

def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)][ : :-1]

def tokenize_en(text):
    return [token.text.lower() for token in spacy_en.tokenizer(text)]

def build_vocab(data, min_freq, max_numb):
    counter = {}
    for sent in data:
        for word in sent:
            counter[word] = counter.get(word, 0) + 1
    word_dict = {word: count for word, count in counter.items() if count >= min_freq}
    word_list = sorted(word_dict.items(), key = lambda x: x[1], reverse = True)[:max_numb - 4]

    words = [word for word, count in word_list]
    words.insert(0, SOS_TOKEN)
    words.insert(0, EOS_TOKEN)
    words.insert(0, UNK_TOKEN)
    words.insert(0, PAD_TOKEN)

    vocab = {}
    vocab['id2word'] = {idx: word for idx, word in enumerate(words)}
    vocab['word2id'] = {word: idx for idx, word in enumerate(words)}
    vocab['special'] = {
        'SOS_TOKEN': '<SOS>',
        'EOS_TOKEN': '<EOS>',
        'UNK_TOKEN': '<UNK>',
        'PAD_TOKEN': '<PAD>'
    }
    return vocab

def encode(tokens, vocab, max_seq_len):
    SOS_TOKEN = vocab['special']['SOS_TOKEN']
    EOS_TOKEN = vocab['special']['EOS_TOKEN']
    UNK_TOKEN = vocab['special']['UNK_TOKEN']
    PAD_TOKEN = vocab['special']['PAD_TOKEN']
    tokens = tokens[:max_seq_len - 2]
    tokens = [SOS_TOKEN] + tokens + [EOS_TOKEN] + [PAD_TOKEN] * (max_seq_len - 2 - len(tokens))
    tokens = [vocab['word2id'].get(token) if vocab['word2id'].get(token) else vocab['word2id'].get(UNK_TOKEN) for token in tokens]
    return tokens

def get_loader(option):

    # total

    train_path = os.path.join(option.targets_path, option.train_file)
    valid_path = os.path.join(option.targets_path, option.valid_file)
    test_path  = os.path.join(option.targets_path, option.test_file )

    train_data = read_data(train_path)
    valid_data = read_data(valid_path)
    test_data  = read_data(test_path )

    # source

    train_src_sent = [data[0] for data in train_data]
    valid_src_sent = [data[0] for data in valid_data]
    test_src_sent  = [data[0] for data in test_data ]

    train_src_word = [tokenize_de(sent) for sent in train_src_sent]
    valid_src_word = [tokenize_de(sent) for sent in valid_src_sent]
    test_src_word  = [tokenize_de(sent) for sent in test_src_sent ]

    # target

    train_trg_sent = [data[1] for data in train_data]
    valid_trg_sent = [data[1] for data in valid_data]
    test_trg_sent  = [data[1] for data in test_data ]

    train_trg_word = [tokenize_en(sent) for sent in train_trg_sent]
    valid_trg_word = [tokenize_en(sent) for sent in valid_trg_sent]
    test_trg_word  = [tokenize_en(sent) for sent in test_trg_sent ]

    # vocab

    src_vocab = build_vocab(train_src_word, option.min_freq, option.max_numb)
    trg_vocab = build_vocab(train_trg_word, option.min_freq, option.max_numb)

    # source

    train_src_id = [encode(word, src_vocab, option.max_seq_len) for word in train_src_word]
    valid_src_id = [encode(word, src_vocab, option.max_seq_len) for word in valid_src_word]
    test_src_id  = [encode(word, src_vocab, option.max_seq_len) for word in test_src_word ]

    train_src_tensor = torch.tensor(train_src_id, dtype = torch.long)
    valid_src_tensor = torch.tensor(valid_src_id, dtype = torch.long)
    test_src_tensor  = torch.tensor(test_src_id , dtype = torch.long)

    # target

    train_trg_id = [encode(word, trg_vocab, option.max_seq_len) for word in train_trg_word]
    valid_trg_id = [encode(word, trg_vocab, option.max_seq_len) for word in valid_trg_word]
    test_trg_id  = [encode(word, trg_vocab, option.max_seq_len) for word in test_trg_word ]

    train_trg_tensor = torch.tensor(train_trg_id, dtype = torch.long)
    valid_trg_tensor = torch.tensor(valid_trg_id, dtype = torch.long)
    test_trg_tensor  = torch.tensor(test_trg_id , dtype = torch.long)

    # dataset & dataloader

    train_dataset = torch.utils.data.TensorDataset(train_src_tensor, train_trg_tensor)
    valid_dataset = torch.utils.data.TensorDataset(valid_src_tensor, valid_trg_tensor)
    test_dataset  = torch.utils.data.TensorDataset(test_src_tensor , test_trg_tensor )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = option.batch_size, shuffle = True )
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = option.batch_size, shuffle = False)
    test_loader  = torch.utils.data.DataLoader(test_dataset , batch_size = option.batch_size, shuffle = False)

    # return

    return src_vocab, trg_vocab, train_loader, valid_loader, test_loader

if __name__ == '__main__':
    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    src_vocab, trg_vocab, train_loader, valid_loader, test_loader = get_loader(option)

    # vocab
    print(type(src_vocab), src_vocab.keys()) # <class 'dict'> dict_keys(['id2word', 'word2id', 'special'])
    print(type(trg_vocab), trg_vocab.keys()) # <class 'dict'> dict_keys(['id2word', 'word2id', 'special'])
    print(len(src_vocab['word2id']), len(src_vocab['id2word'])) # 30000 30000
    print(len(trg_vocab['word2id']), len(trg_vocab['id2word'])) # 22716 22716

    # dataloader
    print(len(train_loader.dataset)) # 304341
    print(len(valid_loader.dataset)) # 16907
    print(len(test_loader .dataset)) # 16907
    for mini_batch in train_loader:
        src, trg = mini_batch
        print(src.shape) # (batch_size, max_seq_len)
        print(trg.shape) # (batch_size, max_seq_len)
        break
