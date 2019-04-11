#Res2Net
from __future__ import division
import torch
import torch.nn as nn
import os
from torch.autograd import Variable

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#SEblock is https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
class SEblock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    


class Res2Net(nn.Module):
    def __init__(self, features_size, stride_ = 1, scale = 4, padding_ = 1, groups_ = 1, reduction = 16):
        super(Res2Net,self).__init__()
        #erro for wrong input
        if scale < 2 or features_size % scale:
            print('Error:illegal input for scale or feature size')

        self.divided_features = int(features_size / scale)
        self.conv1 = nn.Conv2d(features_size, features_size, kernel_size=1, stride=stride_, padding=0, groups=groups_)
        self.conv2 = nn.Conv2d(self.divided_features, self.divided_features, kernel_size=3, stride=stride_, padding=padding_, groups=groups_)
        self.convs = nn.ModuleList()
        self.se = SEblock(features_size, reduction)
        for i in range(scale - 2):

            self.convs.append(
                nn.Conv2d(self.divided_features, self.divided_features, kernel_size=3, stride=stride_, padding=padding_, groups=groups_)
            )


    def forward(self, x):
        features_in = x
        conv1_out = self.conv1(features_in)
        y1 = conv1_out[:,0:self.divided_features,:,:]
        fea = self.conv2(conv1_out[:,self.divided_features:2*self.divided_features,:,:])
        features = fea
        for i, conv in enumerate(self.convs):
            pos = (i + 1)*self.divided_features
            divided_feature = conv1_out[:,pos:pos+self.divided_features,:,:]
            fea = conv(fea + divided_feature)
            features = torch.cat([features, fea], dim = 1)

        out = torch.cat([y1, features], dim = 1)
        conv1_out1 = self.conv1(out)
        se_out = self.se(conv1_out1)
        result = features_in + se_out
        return result

if __name__ == "__main__":
    res2net = Res2Net(64,1,4,1,1,16)
    res2net.cuda()
    x = Variable(torch.rand([8, 64, 32, 32]).cuda())
    y = res2net(x)
    print(x.shape)
    print(y.shape)
    print(res2net)
    torch.save(res2net, 'Res2Net.pth')
