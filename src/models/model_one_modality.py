########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.resnet import ResNet18, ResNet34, ResNet50
from src.models.context_modules import get_context_module
from src.models.model_utils import ConvBNAct, Swish, Hswish
from src.models.model import Decoder

debug = False

def logging(mess):
    if debug: print(mess)


class OWSNetwork(nn.Module):
    def __init__(
        self,
        height=480,
        width=640,
        num_classes=37,
        encoder="resnet18",
        encoder_block="BasicBlock",
        channels_decoder=None,  # default: [128, 128, 128]
        pretrained_on_imagenet=True,
        pretrained_dir=None,
        activation="relu",
        input_channels=3,
        encoder_decoder_fusion="add",
        context_module="ppm",
        nr_decoder_blocks=None,  # default: [1, 1, 1]
        weighting_in_encoder="None",
        upsampling="bilinear",
    ):
        super(OWSNetwork, self).__init__()
        if channels_decoder is None:
            channels_decoder = [128, 128, 128]
        if nr_decoder_blocks is None:
            nr_decoder_blocks = [1, 1, 1]

        print(f"channels_decoder: {channels_decoder}")
        print(f"nr_decoder_blocks: {nr_decoder_blocks}")
        print(f"activation: {activation}")

        self.weighting_in_encoder = weighting_in_encoder

        if activation.lower() == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() in ["swish", "silu"]:
            self.activation = Swish()
        elif activation.lower() == "hswish":
            self.activation = Hswish()
        else:
            raise NotImplementedError(
                "Only relu, swish and hswish as "
                "activation function are supported so "
                "far. Got {}".format(activation)
            )

        # encoder
        if encoder == "resnet18":
            self.encoder = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=False,  # pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=input_channels,
            )
        elif encoder == "resnet34":
            self.encoder = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=input_channels,
            )
        elif encoder == "resnet50":
            self.encoder = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation,
                input_channels=input_channels,
            )
        else:
            raise NotImplementedError(
                "Only ResNets as encoder are supported "
                "so far. Got {}".format(activation)
            )

        self.channels_decoder_in = self.encoder.down_32_channels_out

        self.se_layer0 = nn.Identity()
        self.se_layer1 = nn.Identity()
        self.se_layer2 = nn.Identity()
        self.se_layer3 = nn.Identity()
        self.se_layer4 = nn.Identity()

        if encoder_decoder_fusion == "add":
            layers_skip1 = list()
            if self.encoder.down_4_channels_out != channels_decoder[2]:
                layers_skip1.append(
                    ConvBNAct(
                        self.encoder.down_4_channels_out,
                        channels_decoder[2],
                        kernel_size=1,
                        activation=self.activation,
                    )
                )
            self.skip_layer1 = nn.Sequential(*layers_skip1)

            layers_skip2 = list()
            if self.encoder.down_8_channels_out != channels_decoder[1]:
                layers_skip2.append(
                    ConvBNAct(
                        self.encoder.down_8_channels_out,
                        channels_decoder[1],
                        kernel_size=1,
                        activation=self.activation,
                    )
                )
            self.skip_layer2 = nn.Sequential(*layers_skip2)

            layers_skip3 = list()
            if self.encoder.down_16_channels_out != channels_decoder[0]:
                layers_skip3.append(
                    ConvBNAct(
                        self.encoder.down_16_channels_out,
                        channels_decoder[0],
                        kernel_size=1,
                        activation=self.activation,
                    )
                )
            self.skip_layer3 = nn.Sequential(*layers_skip3)

        # context module
        if "learned-3x3" in upsampling:
            print(
                "Notice: for the context module the learned upsampling is "
                "not possible as the feature maps are not upscaled "
                "by the factor 2. We will use nearest neighbor "
                "instead."
            )
            upsampling_context_module = "nearest"
        else:
            upsampling_context_module = upsampling
        self.context_module, channels_after_context_module = get_context_module(
            context_module,
            self.channels_decoder_in,
            channels_decoder[0],
            input_size=(height // 32, width // 32),
            activation=self.activation,
            upsampling_mode=upsampling_context_module,
        )

        # decoders
        print(f"num_classes: {num_classes}")
        self.decoder_ss = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=num_classes,
        )

        self.decoder_ow = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=19#num_classes#=19,#?? ok, because they could be less than eligible
        )

    def forward(self, image):
        # logging("--> OWNetwork fowrward")
        # logging(f"input image.shape:{image.shape}")
        # logging("1- stem layers")
        out = self.encoder.forward_first_conv(image)
        out = self.se_layer0(out)
        # logging(f"after forward_first_conv: {out.shape}")
        out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        # logging(f"after max_pool2d: {out.shape}")

        # logging(f"2- block 1 {out.shape}")
        # block 1
        out = self.encoder.forward_layer1(out)
        out = self.se_layer1(out)
        skip1 = self.skip_layer1(out)

        # block 2
        # logging(f"3- block 2 {out.shape}")
        out = self.encoder.forward_layer2(out)
        out = self.se_layer2(out)
        skip2 = self.skip_layer2(out)

        # block 3
        # logging(f"4- block 3 {out.shape}")
        out = self.encoder.forward_layer3(out)
        out = self.se_layer3(out)
        skip3 = self.skip_layer3(out)

        # block 4
        # logging(f"5- block 4 {out.shape}")
        out = self.encoder.forward_layer4(out)
        out = self.se_layer4(out)

        # logging(f"6- context_module {out.shape}")
        out = self.context_module(out)

        outs = [out, skip3, skip2, skip1]

        # logging(f"7- decoder ss {out.shape}")
        pred_scales = self.decoder_ss(enc_outs=outs)

        # logging(f"8 - decoder ow")
        ow_res = self.decoder_ow(enc_outs=outs)

        # logging(f"-- OWNetwork  output: pred_scales.shape={pred_scales.shape} ow_res.shape={ow_res.shape} ")

        return pred_scales, ow_res


def main():
    """
    Useful to check if model is built correctly.
    """
    model = OWSNetwork()
    print(model)

    model.eval()
    rgb_image = torch.randn(1, 3, 480, 640)
    # rgb_image = torch.randn(1, 3, 1080, 1920) # dont' work cause last skip dim has dim 135 while corresponding decoding one is 136!!
    # rgb_image = torch.randn(1, 3, 520, 1024) # neither the default oness

    from torch.autograd import Variable

    inputs_rgb = Variable(rgb_image)
    with torch.no_grad():
        output = model(inputs_rgb)


if __name__ == "__main__":
    main()
