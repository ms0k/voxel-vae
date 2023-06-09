"""
@inproceedings{islam2019brain,
  title={Brain tumor segmentation and survival prediction using 3D attention UNet},
  author={Islam, Mobarakol and Vibashan, VS and Jose, V Jeya Maria and Wijethilake, Navodini and Utkarsh, Uppal and Ren, Hongliang},
  booktitle={International MICCAI Brainlesion Workshop},
  pages={262--272},
  year={2019},
  organization={Springer}
}
https://arxiv.org/pdf/2104.00985.pdf
https://github.com/mobarakol/3D_Attention_UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SCA3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel))
        self.spatial_se = nn.Conv3d(channel, 1, kernel_size=1,
                                    stride=1, padding=0, bias=False)

    def forward(self, x):
        bahs, chs, _, _, _ = x.size()
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = torch.sigmoid(self.channel_excitation(chn_se).view(bahs, chs, 1, 1, 1))
        chn_se = torch.mul(x, chn_se)
        spa_se = torch.sigmoid(self.spatial_se(x))
        spa_se = torch.mul(x, spa_se)
        net_out = spa_se + x + chn_se
        return net_out
