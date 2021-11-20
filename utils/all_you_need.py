import tqdm
import torch

def save_checkpoint(save_path, model, optim, epoch):
    checkpoint = {
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(load_path, model, optim):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])
    return checkpoint['epoch']

def train(module_id, model, loader, criterion, optimizer, device, clip):
    model.train()
    epoch_loss = 0
    for mini_batch in tqdm.tqdm(loader):
        mini_batch = [data_item.to(device) for data_item in mini_batch]
        if module_id in [1, 2, 3]:
            source, target = mini_batch
            source, target = source.transpose(0, 1), target.transpose(0, 1)
            output = model(source, target)
        if module_id in [4]:
            source, source_length, target, target_length = mini_batch
            source, target = source.transpose(0, 1), target.transpose(0, 1)
            output = model(source, source_length, target)
        output = output[1:].reshape(-1, output.shape[-1])
        target = target[1:].reshape(-1)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def valid(module_id, model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for mini_batch in tqdm.tqdm(loader):
            mini_batch = [data_item.to(device) for data_item in mini_batch]
            if module_id in [1, 2, 3]:
                source, target = mini_batch
                source, target = source.transpose(0, 1), target.transpose(0, 1)
                output = model(source, target, 0)
            if module_id in [4]:
                source, source_length, target, target_length = mini_batch
                source, target = source.transpose(0, 1), target.transpose(0, 1)
                output = model(source, source_length, target, 0)
            output = output[1:].reshape(-1, output.shape[-1])
            target = target[1:].reshape(-1)
            loss = criterion(output, target)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)
