import torchhaarfeatures
import torch
import torch.nn as nn

from lib.layers import Conv3dBNAct


def init_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # nn.init.normal_(m.weight)
        # some layers may not have bias, so skip if this isnt found
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class ECONetFCNLearnedFeatures(nn.Module):
    def __init__(
        self,
        feat_kernel_size=6,
        feat_num_filters=64,
        hidden_layers=[32, 16],
        num_classes=2,
        feat_padding=None,
        use_bn=True,
        activation=nn.ReLU,
        dropout=0.1,
    ):
        super().__init__()

        # add learned feature extractor
        self.featureextractor = Conv3dBNAct(
            in_channels=1,
            out_channels=feat_num_filters,
            kernel_size=feat_kernel_size,
            stride=1,
            padding=feat_padding,
            use_bn=True,
            activation=activation,
            dropout=dropout,
        )
        in_channels_current_layer = feat_num_filters

        # add hidden mlp layers
        self.mlp_layers = []
        for hlayer in hidden_layers:
            self.mlp_layers.append(
                Conv3dBNAct(
                    in_channels=in_channels_current_layer,
                    out_channels=hlayer,
                    kernel_size=1,
                    stride=1,
                    use_bn=use_bn,
                    activation=activation,
                    dropout=dropout,
                )
            )
            in_channels_current_layer = hlayer

        # add final layer
        self.mlp_layers.append(
            nn.Conv3d(
                in_channels=in_channels_current_layer,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
            )
        )
        self.mlp_layers = nn.Sequential(*self.mlp_layers)
        self.featureextractor.apply(init_weights)
        self.mlp_layers.apply(init_weights)

    def forward(self, x, skip_feat=False, skip_mlp=False):
        if not skip_feat:
            x = self.featureextractor(x)

        if not skip_mlp:
            x = self.mlp_layers(x)
        return x


class ECONetFCNHaarFeatures(nn.Module):
    def __init__(
        self,
        kernel_size=6,
        hidden_layers=[32, 16],
        num_classes=2,
        haar_padding=None,
        use_bn=True,
        activation=nn.ReLU,
        dropout=0.1,
    ):
        super().__init__()

        # add haar-like feature extractor
        self.haarfeatureextactor = torchhaarfeatures.HaarFeatures3d(
            kernel_size=kernel_size,
            padding=haar_padding,
            stride=1,
            padding_mode="zeros",
        )
        in_channels_current_layer = self.haarfeatureextactor.out_channels

        # add hidden mlp layers
        self.mlp_layers = []
        for hlayer in hidden_layers:
            self.mlp_layers.append(
                Conv3dBNAct(
                    in_channels=in_channels_current_layer,
                    out_channels=hlayer,
                    kernel_size=1,
                    stride=1,
                    use_bn=use_bn,
                    activation=activation,
                    dropout=dropout,
                )
            )
            in_channels_current_layer = hlayer

        # add final layer
        self.mlp_layers.append(
            nn.Conv3d(
                in_channels=in_channels_current_layer,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
            )
        )
        self.mlp_layers = nn.Sequential(*self.mlp_layers)
        self.mlp_layers.apply(init_weights)

    def forward(self, x, skip_feat=False, skip_mlp=False):
        if not skip_feat:
            x = self.haarfeatureextactor(x)

        if not skip_mlp:
            x = self.mlp_layers(x)
        return x


if __name__ == "__main__":
    print(ECONetFCNHaarFeatures())
    print(ECONetFCNLearnedFeatures())
