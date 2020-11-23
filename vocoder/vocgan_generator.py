"""
	[VocGAN] Generator

		this source code is implemenation of the modified-VocGAN from rishikksh20
		git repository: https://github.com/rishikksh20/VocGAN
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_WAV_VALUE = 32768.0


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class ResStack(nn.Module):
    def __init__(self, channel, dilation=1):
        super(ResStack, self).__init__()

        self.block = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(dilation),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=3, dilation=dilation)),
                nn.LeakyReLU(0.2),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1)),
            )


        self.shortcut = nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1))


    def forward(self, x):
        return self.shortcut(x) + self.block(x)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.block[2])
        nn.utils.remove_weight_norm(self.block[4])
        nn.utils.remove_weight_norm(self.shortcut)


# Modified VocGAN
class Generator(nn.Module):
    def __init__(self, mel_channel, n_residual_layers, ratios=[4, 4, 2, 2, 2, 2], mult=256, out_band=1):
        super(Generator, self).__init__()
        self.mel_channel = mel_channel

        self.start = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(mel_channel, mult * 2, kernel_size=7, stride=1))
        )
        r = ratios[0]
        self.upsample_1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(mult * 2, mult,
                                                    kernel_size=r * 2, stride=r,
                                                    padding=r // 2 + r % 2,
                                                    output_padding=r % 2)
                                 )
        )
        self.res_stack_1 = nn.Sequential(*[ResStack(mult, dilation=3 ** j) for j in range(n_residual_layers)])

        r = ratios[1]
        mult = mult // 2
        self.upsample_2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(mult * 2, mult,
                                                    kernel_size=r * 2, stride=r,
                                                    padding=r // 2 + r % 2,
                                                    output_padding=r % 2)
                                 )
        )
        self.res_stack_2 = nn.Sequential(*[ResStack(mult, dilation=3 ** j) for j in range(n_residual_layers)])

        r = ratios[2]
        mult = mult // 2
        self.upsample_3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(mult * 2, mult,
                                                    kernel_size=r * 2, stride=r,
                                                    padding=r // 2 + r % 2,
                                                    output_padding=r % 2)
                                 )
        )

        self.skip_upsample_1 = nn.utils.weight_norm(nn.ConvTranspose1d(mel_channel, mult,
                                                                       kernel_size=64, stride=32,
                                                                       padding=16,
                                                                       output_padding=0)
                                                    )
        self.res_stack_3 = nn.Sequential(*[ResStack(mult, dilation=3 ** j) for j in range(n_residual_layers)])



        r = ratios[3]
        mult = mult // 2
        self.upsample_4 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(mult * 2, mult,
                                                    kernel_size=r * 2, stride=r,
                                                    padding=r // 2 + r % 2,
                                                    output_padding=r % 2)
                                 )
        )

        self.skip_upsample_2 = nn.utils.weight_norm(nn.ConvTranspose1d(mel_channel, mult,
                                                                       kernel_size=128, stride=64,
                                                                       padding=32,
                                                                       output_padding=0)
                                                    )
        self.res_stack_4 = nn.Sequential(*[ResStack(mult, dilation=3 ** j) for j in range(n_residual_layers)])


        r = ratios[4]
        mult = mult // 2
        self.upsample_5 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(mult * 2, mult,
                                                    kernel_size=r * 2, stride=r,
                                                    padding=r // 2 + r % 2,
                                                    output_padding=r % 2)
                                 )
        )

        self.skip_upsample_3 = nn.utils.weight_norm(nn.ConvTranspose1d(mel_channel, mult,
                                                                       kernel_size=256, stride=128,
                                                                       padding=64,
                                                                       output_padding=0)
                                                    )
        self.res_stack_5 = nn.Sequential(*[ResStack(mult, dilation=3 ** j) for j in range(n_residual_layers)])


        r = ratios[5]
        mult = mult // 2
        self.upsample_6 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(mult * 2, mult,
                                                    kernel_size=r * 2, stride=r,
                                                    padding=r // 2 + r % 2,
                                                    output_padding=r % 2)
                                 )
        )

        self.skip_upsample_4 = nn.utils.weight_norm(nn.ConvTranspose1d(mel_channel, mult,
                                                                       kernel_size=512, stride=256,
                                                                       padding=128,
                                                                       output_padding=0)
                                                    )
        self.res_stack_6 = nn.Sequential(*[ResStack(mult, dilation=3 ** j) for j in range(n_residual_layers)])

        self.out = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(mult, out_band, kernel_size=7, stride=1)),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, mel):
        mel = (mel + 5.0) / 5.0  # roughly normalize spectrogram
        # Mel Shape [B, num_mels, T] -> torch.Size([3, 80, 10])
        x = self.start(mel)  # [B, dim*2, T] -> torch.Size([3, 512, 10])

        x = self.upsample_1(x)
        x = self.res_stack_1(x)  # [B, dim, T*4] -> torch.Size([3, 256, 40])

        x = self.upsample_2(x)
        x = self.res_stack_2(x)  # [B, dim/2, T*16] -> torch.Size([3, 128, 160])

        x = self.upsample_3(x)
        x = x + self.skip_upsample_1(mel)
        x = self.res_stack_3(x)  # [B, dim/4, T*32] -> torch.Size([3, 64, 320])

        x = self.upsample_4(x)
        x = x + self.skip_upsample_2(mel)
        x = self.res_stack_4(x)  # [B, dim/8, T*64] -> torch.Size([3, 32, 640])

        x = self.upsample_5(x)
        x = x + self.skip_upsample_3(mel)
        x = self.res_stack_5(x)  # [B, dim/16, T*128] -> torch.Size([3, 16, 1280])

        x = self.upsample_6(x)
        x = x + self.skip_upsample_4(mel)
        x = self.res_stack_6(x)  # [B, dim/32, T*256] -> torch.Size([3, 8, 2560])

        out = self.out(x)  # [B, 1, T*256] -> torch.Size([3, 1, 2560])

        return out 

    def eval(self, inference=False):
        super(Generator, self).eval()

        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)


    def infer(self, mel):
        hop_length = 256
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        zero = torch.full((1, self.mel_channel, 10), -11.5129).to(mel.device)
        mel = torch.cat((mel, zero), dim=2)

        audio = self.forward(mel)
        return audio




