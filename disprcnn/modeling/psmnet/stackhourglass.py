import math

import torch.utils.data
from .submodule import *


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class PSMNet(nn.Module):
    def __init__(self, maxdisp, mindisp=0, input_size=224, is_module=False,
                 feature_level=1, single_modal_weight_average=False, conv_layers=(),
                 use_disparity_regression=True
                 ):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.mindisp = mindisp
        self.feature_extraction = feature_extraction()
        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, inputs):
        if isinstance(inputs, dict):
            left, right = inputs['left'], inputs['right']
        elif len(inputs) == 2:
            left, right = inputs
        bsz, _, H, W = left.shape
        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)

        _, C, Hp, Wp = refimg_fea.shape
        # matching
        cost = torch.zeros(bsz, C * 2, (self.maxdisp - self.mindisp) // 4, Hp, Wp).float().to(refimg_fea.device)
        for i in range(self.mindisp // 4, self.maxdisp // 4):
            if i < 0:
                cost[:, :C, i - self.mindisp // 4, :, :i] = refimg_fea[:, :, :, :i]
                cost[:, C:, i - self.mindisp // 4, :, :i] = targetimg_fea[:, :, :, -i:]
            elif i > 0:
                cost[:, :C, i - self.mindisp // 4, :, i:] = refimg_fea[:, :, :, i:]
                cost[:, C:, i - self.mindisp // 4, :, i:] = targetimg_fea[:, :, :, :-i]
            else:
                cost[:, :C, i - self.mindisp // 4, :, :] = refimg_fea
                cost[:, C:, i - self.mindisp // 4, :, :] = targetimg_fea
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2
        if self.training:
            cost1 = F.interpolate(cost1, [self.maxdisp - self.mindisp, H, W], mode='trilinear',
                                  align_corners=True)
            cost2 = F.interpolate(cost2, [self.maxdisp - self.mindisp, H, W], mode='trilinear',
                                  align_corners=True)

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparityregression(pred1, self.maxdisp, self.mindisp)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparityregression(pred2, self.maxdisp, self.mindisp)

            cost3 = F.interpolate(cost3, [self.maxdisp - self.mindisp, H, W], mode='trilinear',
                                  align_corners=True)
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            # For your information: This formulation 'softmax(c)' learned "similarity"
            # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
            # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
            pred3 = disparityregression(pred3, self.maxdisp, self.mindisp)
            return pred1, pred2, pred3
        else:
            cost3 = F.interpolate(cost3, [self.maxdisp - self.mindisp, H, W], mode='trilinear',
                                  align_corners=True)
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparityregression(pred3, self.maxdisp, self.mindisp)
            return pred3
