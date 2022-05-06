import time

import torch
import torch.nn as nn
import math
import torch.nn.functional as m_func
import numpy as np
import cv2


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    """
    Hard sigmoid function introduced in "Searching for MobileNetV3"
    """
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """
     Hard swish function introduced in "Squeeze-and-Excitation Networks"
     """
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation block introduced in "Squeeze-and-Excitation Networks"
    """
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    """
    Inverted residual block introduced in "Searching for MobileNetV3"
    """
    def __init__(self, inp, hidden_dim, out, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == out

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if use_hs else nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if use_hs else nn.ReLU(inplace=True),

            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Identity(),
            # pw-linear
            nn.Conv2d(hidden_dim, out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out),
        )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class SkipBlock(nn.Module):
    """
    Skip Block: simple module designed to connect together the blocks with the different spatial sizes
    """
    def __init__(self, inp, hidden_dim, out, kernel_size, stride, size):
        super(SkipBlock, self).__init__()
        assert stride in [1, 2]
        self.size = size
        self.identity = stride == 1 and inp == out

        self.core_block = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            # pw-linear
            nn.Conv2d(hidden_dim, out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out),
        )

    def forward(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, self.size)
        if self.identity:
            return x + self.core_block(x)
        else:
            return self.core_block(x)


class SkipNet(nn.Module):
    """
    SkipNet model introduced in "Bias Loss for Mobile Neural Networks"
    """
    def __init__(self, cfgs,cfgs_skip,num_classes, mode,  width_mult=1.):
        super(SkipNet, self).__init__()

        # configurations of inverted residual and skip blocks
        self.cfgs = cfgs
        self.cfgs_skipblocks = cfgs_skip

        assert mode in ['large', 'small']

        # building inverted residual and skip blocks
        block_inv = InvertedResidual
        block_skip = SkipBlock

        # first layer plus base inverted residual block
        input_channel = 32
        input_channel = _make_divisible(input_channel * width_mult, 8)

        self.first_layer = conv_3x3_bn(3, input_channel, 2)
        self.base_block = block_inv(input_channel, input_channel, input_channel, 3, 1, 0, 1)

        # inverted residual blocks
        self.blocks = nn.ModuleList([])
        features = []
        for k, t, output_channel, use_se, use_hs, s, make_block in self.cfgs:
            output_channel = _make_divisible(output_channel * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            features.append(block_inv(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel

            if make_block:
                if len(self.blocks)<2:
                    features.append(nn.Dropout(0.1))
                else:
                    features.append(nn.Dropout(0.1))
                self.blocks.append(nn.Sequential(*features))
                features = []

        # skip blocks
        skip1_cfg = cfgs_skip[0]
        exp_size_int = _make_divisible(skip1_cfg[0] * skip1_cfg[1], 8)
        output_channel = _make_divisible(skip1_cfg[2] * width_mult, 8)
        self.skip1 = block_skip(inp=skip1_cfg[0], hidden_dim=exp_size_int, out=output_channel, kernel_size=skip1_cfg[3], stride=skip1_cfg[4], size=skip1_cfg[5])

        skip2_cfg = cfgs_skip[1]
        exp_size_int = _make_divisible(skip2_cfg[0] * skip2_cfg[1], 8)
        output_channel = _make_divisible(skip2_cfg[2] * width_mult, 8)
        self.skip2 = block_skip(inp=skip2_cfg[0], hidden_dim=exp_size_int, out=output_channel, kernel_size=skip2_cfg[3],
                                stride=skip2_cfg[4], size=skip2_cfg[5])

        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # output channels of fully-connected layer
        output_channel = {'large': 320, 'small': 320}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]

        # classification block
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.0),

        )
        self.liner=  nn.Linear(output_channel, num_classes)

        # self.liner = nn.Sequential(
        #     nn.Linear(exp_size, output_channel),
        #     h_swish(),
        #     nn.Dropout(0.0),
        #     nn.Linear(output_channel, 3),
        # )

        self._initialize_weights()

    def forward(self, x):
        x = self.first_layer(x)
        x_base = self.base_block(x)

        x = self.blocks[0](x_base)

        x_skip1 = self.skip1(x_base)
        x = self.blocks[1](x + x_skip1)

        x_skip2 = self.skip2(x_base)
        x = self.blocks[2](x + x_skip2)

        x = self.blocks[3](x)

        x_features = self.conv(x)
        x = self.avgpool(x_features)
        x = x.view(x.size(0), -1)
        cat_out = self.classifier(x)
        out=self.liner(cat_out)
        # cont_out = self.liner(x)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def get_skipnet(num_classes, mode='large', width=1.0,skip_w=8):
    """
    Constructs a SkipNet model,with
     cfgs - default configurations for the inverted residual blocks
     cfgs_skip - default configurations for the skip blocks
    """
    cfgs = [
        # kernel,  exp_size,  out_channel, SE, HS, stride, make_block

        [3,   3,  24, 0, 0, 2, False],
        [3,   3,  24, 0, 0, 1, False],
        [5,   3,  40, 1, 0, 2, False],
        [5,   3,  40, 1, 0, 1, False],
        [5,   3,  40, 1, 1, 1, True],

        [3,  3,  80, 0, 1, 2, False],
        [3, 2.5,  80, 0, 1, 1, False],
        [3, 2.3,  80, 0, 1, 1, False],
        [3, 2.3,  80, 0, 1, 1, True],

        [3,   3, 112, 1, 1, 1, False],
        [3,   3, 112, 1, 1, 1, True],
        [5,   3, 160, 1, 1, 2, False],
        [5,   3, 160, 1, 1, 1, False],
        [5,   3, 160, 1, 1, 1, True]
    ]

    cfgs_skip = [
        #input_channel, exp_size, out_channel, kernel, stride, feat_width
        [32, 6, 40, 5, 1, skip_w*2],
        [32, 2.3, 80, 3, 1, skip_w]
    ]
    return SkipNet(cfgs, cfgs_skip, num_classes, mode=mode, width_mult=width)

class Skipnet_Class():

    def __init__(self,model_path,num_classes=6):
        self.model = get_skipnet(num_classes=num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes=num_classes
        # state_dict = torch.load(r"0.9926_14.pth")
        if model_path!="":
            state_dict = torch.load(model_path)
        # state_dict = torch.load(r"0.9976_34.pth")
            self.model.load_state_dict(state_dict)
        # self.model.cuda()
        self.model.eval()
        self.model = self.model.to(self.device)

    def get_cls(self,imgs):

        inputs=[]
        for img in imgs:
            try:
                img = cv2.resize(img, (64, 64))
                img=img.astype(np.float32)
                img=img.transpose(2,0,1)
                d_t_trans_2 = torch.from_numpy(img)
                inputs.append(d_t_trans_2.unsqueeze(0))
            except Exception as e:
                print(e)

        d_t_trans=torch.cat(inputs,0)

        d_t_trans=d_t_trans.to(self.device)
        time1 = time.time()
        output_o = self.model(d_t_trans)
        # output_o = output_o.contiguous().view(output_o.size(0), self.num_classes, 1)
        print(output_o)
        outputs = m_func.softmax(output_o, dim=1)
        scores, preds = torch.max(outputs.data, 1)

        if torch.cuda.is_available():
            return preds.cpu().data.numpy(),scores.cpu().data.numpy(),output_o.cpu().data.numpy()
        return preds.data.numpy(),scores.data.numpy(),output_o.data.numpy()
if __name__ == '__main__':

    model = get_skipnet(num_classes=512,skip_w=8)

    model.eval()

    dummy_input1 = torch.randn(1, 3, 128, 128)#.cuda()

    cat_out, cont_out = model(dummy_input1)

    print(cat_out.size(), cont_out.size())

