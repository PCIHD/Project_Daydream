'''
layers and blocks file
'''
import torch
from torch import nn
from torch.nn import Functional as F
import numpy as np


class DiscriminatorFeature(nn.Module):
    def __init__(self, input_nc, opt, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False,
                 getIntermFeat=False):
        super(DiscriminatorFeature, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.ngpu = len(opt.gpu_ids)

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

        self.apply(init_weights)

    def forward(self, segmentation, attribute):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            input = torch.cat((segmentation, attribute), 1)

            if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                img_feat = nn.parallel.data_parallel(self.model, input, range(self.ngpu))
            else:
                img_feat = self.model(input)

            return img_feat


class SGNNLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, opt, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False,
                 getIntermFeat=False):
        super(SGNNLayerDiscriminator, self).__init__()
        self.ConditionFeature = DiscriminatorFeature(input_nc=48, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d,
                                                     use_sigmoid=False,
                                                     getIntermFeat=False)
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.ngpu = len(opt.gpu_ids)

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

        self.classifier = nn.Sequential(
            nn.Conv2d(512 * 2, 512, 1, stride=(1, 1), padding=0),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw)
        )

        self.apply(init_weights)

    def forward(self, input, segmentation, attribute):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            attribute = attribute.unsqueeze(-1).unsqueeze(-1)
            attribute = attribute.repeat(1, 1, input.size(2), input.size(3))

            if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                img_feat = nn.parallel.data_parallel(self.model, input, range(self.ngpu))
            else:
                img_feat = self.model(input)

            cond_feat = self.ConditionFeature(segmentation, attribute)
            fusion = torch.cat((img_feat, cond_feat), dim=1)
            if isinstance(fusion.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                classout = nn.parallel.data_parallel(self.classifier, fusion, range(self.ngpu))
            else:
                classout = self.classifier(fusion)

            return classout


class SGNResidualBlock(nn.Module):
    def __init__(self):
            super(SGNResidualBlock, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(512 + 40, 512, 3, padding=1, bias=False),
                nn.InstanceNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1, bias=False),
                nn.InstanceNorm2d(512)
            )

    def forward(self, seg_feat, att):
            att = att.unsqueeze(-1).unsqueeze(-1)
            att = att.repeat(1, 1, seg_feat.size(2), seg_feat.size(3))
            fusion = torch.cat((seg_feat, att), dim=1)
            return F.relu(seg_feat + self.encoder(fusion))


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            m.weight.data.normal_(std=0.02)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d) and m.affine:
        if m.weight.requires_grad:
            m.weight.data.normal_(1, 0.02)
        if m.bias.requires_grad:
            m.bias.data.fill_(0)
