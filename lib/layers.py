import math
import numbers
from typing import Optional

import torch
import torch.nn as nn
from monai.losses.dice import DiceLoss
from torch.nn.modules.loss import _Loss


class Conv3dBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding="same",
        use_bn=False,
        activation=nn.ReLU,
        dropout=0.2,
    ):

        modules = []
        modules.append(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )
        if use_bn:
            modules.append(nn.BatchNorm3d(num_features=out_channels))
        if activation:
            modules.append(activation())
        if dropout:
            modules.append(nn.Dropout(p=dropout))

        super().__init__(*modules)


# Help on Gaussian Smoother from:
# https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/8
def initialise_gaussian_weights(channels, kernel_size, sigma, dims):
    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * dims
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dims

    # The gaussian kernel is the product of the
    # gaussian function of each dimension.
    kernel = 1
    meshgrids = torch.meshgrid(
        [torch.arange(size, dtype=torch.float32) for size in kernel_size]
    )
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= (
            1
            / (std * math.sqrt(2 * math.pi))
            * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
        )

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
    return kernel


class GaussianSmoothing2d(nn.modules.Conv2d):
    def __init__(
        self,
        in_channels,
        kernel_size,
        sigma,
        padding="same",
        stride=1,
        padding_mode="zeros",
    ):
        gausssian_weights = initialise_gaussian_weights(
            channels=in_channels, kernel_size=kernel_size, sigma=sigma, dims=2
        )

        out_channels = gausssian_weights.shape[0]

        super(GaussianSmoothing2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=in_channels,
            bias=False,
            padding_mode=padding_mode,
        )

        # update weights
        # help from: https://discuss.pytorch.org/t/how-do-i-pass-numpy-array-to-conv2d-weight-for-initialization/56595/3
        with torch.no_grad():
            haar_weights = gausssian_weights.float().to(self.weight.device)
            self.weight.copy_(haar_weights)


class GaussianSmoothing3d(nn.modules.Conv3d):
    def __init__(
        self,
        in_channels,
        kernel_size,
        sigma,
        padding="same",
        stride=1,
        padding_mode="zeros",
    ):
        gausssian_weights = initialise_gaussian_weights(
            channels=in_channels, kernel_size=kernel_size, sigma=sigma, dims=3
        )

        out_channels = gausssian_weights.shape[0]

        super(GaussianSmoothing3d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=in_channels,
            bias=False,
            padding_mode=padding_mode,
        )

        # update weights
        # help from: https://discuss.pytorch.org/t/how-do-i-pass-numpy-array-to-conv2d-weight-for-initialization/56595/3
        with torch.no_grad():
            haar_weights = gausssian_weights.float().to(self.weight.device)
            self.weight.copy_(haar_weights)


class MyDiceCELoss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=ce_weight,
            reduction=reduction,
        )
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def ce(self, input, target):
        """
        Compute CrossEntropy loss for the input and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch == n_target_ch:
            # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            target = torch.argmax(target, dim=1)
        else:
            target = torch.squeeze(target, dim=1)
        target = target.long()
        return self.cross_entropy(input, target)

    def forward(self, input, target):
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same."
            )

        dice_loss = self.dice(input, target)
        ce_loss = self.ce(input, target)
        if len(dice_loss.shape) < len(ce_loss.shape):
            dice_loss = dice_loss.view(*dice_loss.shape + ce_loss.shape[2:])
        total_loss = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        return total_loss
