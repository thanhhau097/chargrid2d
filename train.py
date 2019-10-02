import torch
import numpy as np

from dataloader import ChargridDataloader
from loss import ChargridLoss
from model import Chargrid2D
# from metrics import


def train():
    N_EPOCHS = 10
    best_loss = np.infty

    dataloader = ChargridDataloader(root='data/', image_size=512, batch_size=4)
    loss_fn = ChargridLoss()
    model = Chargrid2D(input_channels=3, n_classes=10)
    optimizer = torch.optim.Adam(model.params(), lr=0.001)

    for epoch in range(N_EPOCHS):
        print('Epoch {}/{}'.format(epoch, N_EPOCHS - 1))
        print('-' * 10)

        # -------- TRAIN -------
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(dataloader):
            # we need to get gt_seg, gt_boxmask, gt_boxcoord
            img, mask, boxes, lbl_boxes = batch

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            pred_seg, pred_boxmask, pred_boxcoord = model(img)
            loss = loss_fn(pred_seg, pred_boxmask, pred_boxcoord, ...)  # TODO
            epoch_loss += loss.item()  # loss is mean loss of batch
            print("Step", i, 'loss =', loss.item())

            # backward
            loss.backward()
            optimizer.step()

        # -------- EVALUATION -------
        model.eval()
        # TODO: update evaluation


if __name__ == '__main__':
    train()
