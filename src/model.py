import torch
from torch import sigmoid
from torch.nn import Conv2d, MaxPool2d, ConvTranspose2d, Module
from torch.nn.functional import relu
import torch.nn.functional as F


class UNet(Module):
    def __init__(self, n_class, input_channels=3, conv_ks=3, d=32, p=1, num_layers=4):
        super(UNet, self).__init__()

        self.num_layers = num_layers

        # Encoder
        self.encoder_convs = []
        self.encoder_pools = []
        in_channels = input_channels
        for i in range(num_layers):
            setattr(
                self,
                f"e{i}1",
                Conv2d(in_channels, d * (2**i), kernel_size=conv_ks, padding=p),
            )
            setattr(
                self,
                f"e{i}2",
                Conv2d(d * (2**i), d * (2**i), kernel_size=conv_ks, padding=p),
            )
            self.encoder_convs.append(
                (getattr(self, f"e{i}1"), getattr(self, f"e{i}2"))
            )
            if i < num_layers - 1:
                setattr(self, f"pool{i+1}", MaxPool2d(kernel_size=2, stride=2))
                self.encoder_pools.append(getattr(self, f"pool{i+1}"))
            in_channels = d * (2**i)

        # Decoder
        self.decoder_convs = []
        self.upconvs = []
        for i in range(num_layers - 1, 0, -1):
            setattr(
                self,
                f"upconv{i}",
                ConvTranspose2d(
                    d * (2**i), d * (2 ** (i - 1)), kernel_size=2, stride=2
                ),
            )
            setattr(
                self,
                f"d{i}1",
                Conv2d(
                    d * (2**i),
                    d * (2 ** (i - 1)),
                    kernel_size=conv_ks,
                    padding=p,
                ),
            )
            setattr(
                self,
                f"d{i}2",
                Conv2d(
                    d * (2 ** (i - 1)),
                    d * (2 ** (i - 1)),
                    kernel_size=conv_ks,
                    padding=p,
                ),
            )
            self.upconvs.append(getattr(self, f"upconv{i}"))
            self.decoder_convs.append(
                (getattr(self, f"d{i}1"), getattr(self, f"d{i}2"))
            )

        # Output layer
        self.outconv = Conv2d(d, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        encoder_outputs = []
        for i in range(self.num_layers):
            conv1, conv2 = self.encoder_convs[i]
            x = relu(conv1(x))
            x = relu(conv2(x))
            encoder_outputs.append(x)
            if i < self.num_layers - 1:
                x = self.encoder_pools[i](x)

        # Decoder
        for i in range(self.num_layers - 1):
            upconv = self.upconvs[i]
            conv1, conv2 = self.decoder_convs[i]
            x = upconv(x)
            x = torch.cat((x, encoder_outputs[-i - 2]), dim=1)
            x = relu(conv1(x))
            x = relu(conv2(x))

        # Output layer
        out = self.outconv(x)

        return out


class DiceBCELoss(Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            inputs.sum() + targets.sum() + smooth
        )
        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


def IoU(inputs, targets, smooth=1):
    # comment out if your model contains a sigmoid or equivalent activation layer
    inputs = sigmoid(inputs)

    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # intersection is equivalent to True Positive count
    # union is the mutually inclusive area of all labels & predictions
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection

    score = (intersection + smooth) / (union + smooth)

    return score


def get_model(start_channel, num_layers):
    model = UNet(
        n_class=1,
        input_channels=3,
        conv_ks=3,
        d=start_channel,
        p=1,
        num_layers=num_layers,
    ).cuda()
    return model


def get_essentials(model, lr):
    criterion = DiceBCELoss().cuda()
    optimizer = torch.optim.Adam(
        [parameters for parameters in model.parameters() if parameters.requires_grad],
        lr=float(lr),
    )

    return criterion, optimizer


if __name__ == "__main__":
    pass
