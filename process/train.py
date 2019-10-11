import os
import argparse
import time
import shutil

import albumentations as alb
import cv2
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataloader_utils.dataloader import SegDataset
from models.fscnn import FastSCNN
from utils.loss import MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss
from utils.metric import SegmentationMetric

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Fast-SCNN on PyTorch')
    parser.add_argument('--size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--root', type=str, default='./data',
                        help='root data folder')
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=True,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for training (default: 12)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-folder', default='./weights',
                        help='Directory for saving checkpoint models')
    # evaluation only
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    # the parser
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    args.device = device

    print(args)
    return args

class Trainer(object):
    def __init__(self, args):
        self.args = args
        # image transform
        aug = alb.Compose([
            alb.LongestMaxSize(self.args.size + 24, interpolation=0),
            alb.PadIfNeeded(self.args.size + 24, self.args.size + 24, border_mode=cv2.BORDER_CONSTANT),
            alb.RandomCrop(self.args.size, self.args.size, p=0.3),
            alb.Resize(self.args.size, self.args.size, interpolation=0)
        ])

        train_dataset = SegDataset(self.args.root, transform=aug, type='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=1)
        print(len(self.train_loader))

        val_dataset = SegDataset(self.args.root, transform=aug, type='val')
        self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=1)
        # self.train_loader = self.val_loader
        print(len(self.val_loader))

        # create network
        self.model = FastSCNN(in_channels=len(train_dataset.corpus)+1, num_classes=len(train_dataset.target) + 1, aux=self.args.aux)
        # if torch.cuda.device_count() > 1:
        #     self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 2])
        self.model.to(args.device)

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        self.criterion = MixSoftmaxCrossEntropyOHEMLoss(aux=args.aux, aux_weight=args.aux_weight).to(args.device)

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        
        # evaluation metrics
        self.metric = SegmentationMetric(len(train_dataset.target))

        self.best_pred = 0.0

    def train(self):
        start_time = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.model.train()
            epoch_loss = 0.0

            for i, (images, targets) in enumerate(self.train_loader):
                if images.size()[0] == 1:
                    continue
                images = images.to(self.args.device)
                targets = targets.to(self.args.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print('Epoch: [%2d/%2d] || Time: %4.4f sec || Loss: %.4f' % ( \
                epoch, args.epochs, \
                time.time() - start_time, epoch_loss))

            if self.args.no_val:
                # save every epoch
                save_checkpoint(self.model, self.args, is_best=False)
            else:
                self.validation(epoch)

        save_checkpoint(self.model, self.args, is_best=False)

    def validation(self, epoch):
        is_best = False
        self.metric.reset()
        self.model.eval()
        for i, (image, target) in enumerate(self.val_loader):
            image = image.to(self.args.device)

            outputs = self.model(image)
            pred = torch.argmax(outputs[0], 1)
            pred = pred.cpu().data.numpy()
            self.metric.update(pred, target.numpy())
            pixAcc, mIoU = self.metric.get()
            print('Epoch %d, Sample %d, validation pixAcc: %.3f%%, mIoU: %.3f%%' % (
                epoch, i + 1, pixAcc * 100, mIoU * 100))

        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        save_checkpoint(self.model, self.args, is_best)


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_folder)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = 'model.pth'
    save_path = os.path.join(directory, filename)
    torch.save(model.state_dict(), save_path)
    if is_best:
        best_filename = 'best_model.pth'
        best_filename = os.path.join(directory, best_filename)
        torch.save(model.state_dict(), best_filename)

if __name__ == "__main__":
    # python process/train.py --size 512 --batch-size X
    args = parse_args()
    trainer = Trainer(args)
    if args.eval:
        print('Evaluation model: ', args.resume)
        trainer.validation(args.start_epoch)
    else:
        print('Starting Epoch: %d, Total Epochs: %d' % (args.start_epoch, args.epochs))
        trainer.train()