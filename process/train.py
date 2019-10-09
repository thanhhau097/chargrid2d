import os
import argparse
import time
import shutil

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms

from dataloader import ChargridDataloader
from models.fscnn import FastSCNN
from utils.loss import MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss
from utils.metric import SegmentationMetric

class Trainer(object):
    def __init__(self, args):
        self.args = args
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        
        train_dataset = get_segmentation_dataset(args.dataset, split=args.train_split, mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            drop_last=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          shuffle=False)

        # create network
        self.model = get_fast_scnn(dataset=args.dataset, aux=args.aux)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 2])
        self.model.to(args.device)

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        self.criterion = MixSoftmaxCrossEntropyOHEMLoss(aux=args.aux, aux_weight=args.aux_weight,
                                                        ignore_index=-1).to(args.device)

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        
        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)

        self.best_pred = 0.0

    def train(self):
        cur_iters = 0
        start_time = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.model.train()

            for i, (images, targets) in enumerate(self.train_loader):
                images = images.to(self.args.device)
                targets = targets.to(self.args.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                cur_iters += 1
                if cur_iters % 10 == 0:
                    print('Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.8f || Loss: %.4f' % (
                        epoch, args.epochs, i + 1, len(self.train_loader),
                        time.time() - start_time, loss.item()))

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
    filename = '{}_{}.pth'.format(args.model, args.dataset)
    save_path = os.path.join(directory, filename)
    torch.save(model.state_dict(), save_path)
    if is_best:
        best_filename = '{}_{}_best_model.pth'.format(args.model, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)

if __name__ == "__main__":
    pass