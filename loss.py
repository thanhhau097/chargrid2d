import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        # target = target.view(-1, 1)
        target = target.view(-1, input.size()[1])

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class ChargridLoss(nn.Module):
    def __init__(self):
        super(ChargridLoss, self).__init__()
        self.loss_seg_fn = FocalLoss()  # TODO: set hyper parameters
        self.loss_box_mask_fn = FocalLoss()
        self.loss_boxcoord_fn = nn.SmoothL1Loss(size_average=True)
        # https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py

    def forward(self, pred_seg, pred_boxmask, pred_boxcoord, gt_seg, gt_boxmask, gt_boxcoord):
        """

        :param pred_seg: N x C x H x W
        :param pred_boxmask:
        :param pred_boxcoord:
        :param gt_seg: N x H x W x C
        :param gt_boxmask:
        :param gt_boxcoord:
        :return:
        """
        loss_seg = self.loss_seg_fn(pred_seg. gt_seg)
        loss_boxmask = self.loss_box_mask_fn(pred_boxmask, gt_boxmask)

        return loss_seg + loss_boxmask


if __name__ == '__main__':
    x = torch.zeros([1, 10, 64, 64])
    y = torch.zeros([1, 64, 64, 10], dtype=torch.long)
    loss = FocalLoss()
    print(loss(x, y))

