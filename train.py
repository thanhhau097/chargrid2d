import torch
import numpy as np

from dataloader import ChargridDataloader
from loss import ChargridLoss
from model import Chargrid2D
# from metrics import


def train():
    N_EPOCHS = 10
    best_loss = np.infty

    dataloader = ChargridDataloader(root='data/', image_size=128, batch_size=1, validation_split=0.1)
    val_dataloader = dataloader.split_validation()

    loss_fn = ChargridLoss()
    model = Chargrid2D(input_channels=len(dataloader.dataset.corpus) + 1, n_classes=len(dataloader.dataset.target))
    # model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(N_EPOCHS):
        print('Epoch {}/{}'.format(epoch, N_EPOCHS - 1))
        print('-' * 10)

        # -------- TRAIN -------
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(dataloader):
            # we need to get gt_seg, gt_boxmask, gt_boxcoord
            img, mask, boxes, lbl_boxes = batch
            # img, mask, boxes, lbl_boxes = img.cuda(), mask.cuda(), boxes.cuda(), lbl_boxes.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            pred_seg, pred_boxmask, pred_boxcoord = model(img)
            # default ground truth boxmask
            gt_boxmask = torch.ones([pred_boxmask.size()[0], pred_boxmask.size()[2], pred_boxmask.size()[3]]
                                    , dtype=torch.int64)  # TODO: wrong

            mask = mask.type(torch.int64)
            loss = loss_fn(pred_seg, pred_boxmask, pred_boxcoord, mask, gt_boxmask, boxes)
            epoch_loss += loss.item()  # loss is mean loss of batch
            print("Step", i, 'loss =', loss.item())

            # backward
            loss.backward()
            optimizer.step()

        print("Training loss:", epoch_loss / len(dataloader))

        # -------- EVALUATION -------
        model.eval()
        # TODO: update evaluation
        print("START EVALUATION")
        epoch_loss = 0
        for i, batch in enumerate(val_dataloader):
            # we need to get gt_seg, gt_boxmask, gt_boxcoord
            img, mask, boxes, lbl_boxes = batch
            # img, mask, boxes, lbl_boxes = img.cuda(), mask.cuda(), boxes.cuda(), lbl_boxes.cuda()

            # forward
            pred_seg, pred_boxmask, pred_boxcoord = model(img)
            mask = mask.type(torch.int64)

            # default ground truth boxmask
            gt_boxmask = torch.ones([pred_boxmask.size()[0], pred_boxmask.size()[2], pred_boxmask.size()[3]]
                                    , dtype=torch.int64)  # TODO: wrong

            loss = loss_fn(pred_seg, pred_boxmask, pred_boxcoord, mask, gt_boxmask, boxes)
            epoch_loss += loss.item()  # loss is mean loss of batch
            print("Step", i, 'loss =', loss.item())

        print('Validation loss:', epoch_loss / len(val_dataloader))


if __name__ == '__main__':
    train()
