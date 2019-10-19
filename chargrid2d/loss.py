from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from chargrid2d.utils import one_hot

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# based on:
# https://github.com/zhezh/focalloss/blob/master/focalloss.py


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.
    According to [1], the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (Optional[str]): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Examples:
        >>> N = 5  # num_classes
        >>> args = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> loss = kornia.losses.FocalLoss(*args)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float, gamma: Optional[float] = 2.0,
                 reduction: Optional[str] = 'none') -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: torch.Tensor = torch.tensor(gamma).to(device)
        self.reduction: Optional[str] = reduction
        self.eps: float = 1e-6

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1) + self.eps

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)

        # compute the actual focal loss
        weight = torch.pow(torch.tensor(1.) - input_soft,
                           self.gamma.type(input.dtype))
        focal = -self.alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}"
                                      .format(self.reduction))
        return loss


def focal_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        gamma: Optional[float] = 2.0,
        reduction: Optional[str] = 'none') -> torch.Tensor:
    r"""Function that computes Focal loss.
    See :class:`~kornia.losses.FocalLoss` for details.
    """
    return FocalLoss(alpha, gamma, reduction)(input, target)


# TODO: currently only use Segmentation Decoder
class ChargridLoss(nn.Module):
    def __init__(self):
        super(ChargridLoss, self).__init__()
        self.loss_seg_fn = FocalLoss(alpha=0.25, gamma=2, reduction='sum')  # TODO: set hyper parameters
        self.loss_box_mask_fn = FocalLoss(alpha=0.25, gamma=2, reduction='sum')
        self.loss_boxcoord_fn = nn.SmoothL1Loss(size_average=True)
        # https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py

    def forward(self, pred_seg, pred_boxmask, pred_boxcoord, gt_seg, gt_boxmask, gt_boxcoord):
        """

        :param pred_seg: N x C x H x W
        :param pred_boxmask:
        :param pred_boxcoord:
        :param gt_seg: N x H x W
        :param gt_boxmask:
        :param gt_boxcoord:
        :return:
        """
        loss_seg = self.loss_seg_fn(pred_seg, gt_seg)
        # loss_boxmask = self.loss_box_mask_fn(pred_boxmask, gt_boxmask)
        # loss_boxcoord = self.loss_boxcoord_fn(pred_boxcoord, gt_boxcoord)

        return loss_seg  # + loss_boxmask + loss_boxcoord
        #  TODO*: we must follow loss in https://arxiv.org/pdf/1506.01497.pdf


if __name__ == '__main__':
    N = 5
    loss = FocalLoss(0.25, 2, 'mean')
    input = torch.randn(1, N, 3, 5, requires_grad=True)
    target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
    output = loss(input, target)
    print(output)
    output.backward()

