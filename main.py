import os
import subprocess

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

seed = 77

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

from utils.parser import get_parser
from utils.logger import get_logger

parser = get_parser()
option = parser.parse_args()

root_path = 'result'

logs_folder = os.path.join(root_path, 'logs', option.name)
save_folder = os.path.join(root_path, 'save', option.name)

subprocess.run('mkdir -p %s' % logs_folder, shell = True)
subprocess.run('mkdir -p %s' % save_folder, shell = True)

logs_path = os.path.join(logs_folder, 'main.log' )
save_path = os.path.join(save_folder, 'best.ckpt')

logger = get_logger(option.name, logs_path)

from loaders.loader  import get_loader

from modules.module1 import get_module as get_module1
from modules.module2 import get_module as get_module2
from modules.module3 import get_module as get_module3

from utils.all_you_need import train, valid, save_checkpoint, load_checkpoint

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

logger.info('prepare loader')

src_vocab, trg_vocab, train_loader, valid_loader, test_loader = get_loader(option)

logger.info('prepare module')

src_vocab_size = len(src_vocab['word2id'])
trg_vocab_size = len(trg_vocab['word2id'])

assert option.module in range(1, 7)

if option.module == 1:
    seq2seq = get_module1(option, src_vocab_size, trg_vocab_size)
elif option.module == 2:
    seq2seq = get_module2(option, src_vocab_size, trg_vocab_size)
elif option.module == 3:
    seq2seq = get_module3(option, src_vocab_size, trg_vocab_size)

seq2seq = seq2seq.to(device)

logger.info('prepare envs')

optimizer = optim.Adam(seq2seq.parameters())
criterion = nn.CrossEntropyLoss(ignore_index = trg_vocab['word2id'][trg_vocab['special']['PAD_TOKEN']])

logger.info('start training!')

best_valid_loss = float('inf')
for epoch in range(option.num_epochs):
    train_loss = train(seq2seq, train_loader, criterion, optimizer, device, option.grad_clip)
    valid_loss = valid(seq2seq, valid_loader, criterion, device)
    logger.info(
        '[Epoch %d] Train Loss: %f, Valid Loss: %f' %
        (epoch, train_loss, valid_loss)
    )
    if  best_valid_loss > valid_loss:
        best_valid_loss = valid_loss
        save_checkpoint(save_path, seq2seq, optimizer, epoch)

logger.info('start testing!')

load_checkpoint(save_path, seq2seq, optimizer)
test_loss = valid(seq2seq, test_loader, criterion, device)
logger.info('Test Loss: %f' % test_loss)
