import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
from warpctc_pytorch import CTCLoss

from dataloader import Dataset, test_fn, train_fn
from model import CNNCTC
from utils import load_model

torch.backends.cudnn.benchmark = True


def main():
    if args.mode == 'train':
        train_dataset = Dataset(name=args.dataset, mode='train', windows=[24, 32, 40], step=4)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_fn,
                                       shuffle=True, num_workers=args.workers, pin_memory=True)
        train(train_loader)
    if args.mode == 'test':
        test_dataset = Dataset(name=args.dataset, mode='test', windows=[24, 32, 40], step=4)
        test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_fn,
                                      shuffle=False, num_workers=args.workers, pin_memory=True)
        model = load_model(device)
        test(model, test_loader)


def train(train_loader):
    model = CNNCTC(class_num=37).to(device)
    if args.warm_up:
        model = load_model(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_function = CTCLoss(size_average=True, length_average=True).to(device)
    min_loss = np.Inf
    for epoch in range(args.epoch):
        print('%d / %d Epoch' % (epoch, args.epoch))
        epoch_loss = train_epoch(train_loader, model, optimizer, loss_function)
        print(epoch_loss)
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save(model.state_dict(), 'model.bin')
    return model


def train_epoch(train_loader, model, optimizer, loss_function):
    total_loss = 0
    model.train()
    model.mode = 'train'
    for i, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        images = batch.images.to(device)
        labels = batch.labels
        label_lengths = batch.label_lengths
        probs = model(images)
        log_probs = probs.log_softmax(2).to(device)
        prob_lengths = torch.IntTensor([log_probs.size(0)] * args.batch_size)
        loss = loss_function(log_probs, labels, prob_lengths, label_lengths) / args.batch_size
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss


def test(model, test_loader):
    model.eval()
    model.mode = 'test'
    total, correct = 0, 0
    for i, batch in enumerate(tqdm(test_loader)):
        images = batch.images.to(device)
        labels = batch.labels
        out = model(images)
        for actual, label in zip(labels, out):
            if actual == label:
                correct += 1
            total += 1
    print(correct / total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sliding convolution')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--dataset', default='IIIT5K')
    parser.add_argument('--cuda', default=False, type=bool)
    parser.add_argument('--warm-up', default=False, type=bool)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--mode', default='train', type=str)
    args = parser.parse_args()
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    main()
