'''
todo implement model architecture here , Discriminator , generator
'''
import torch
from torch import nn

#Generator class
from Dreamer.Layers import init_weights, SGNNLayerDiscriminator


class SGNGenerator(nn.Module):
    def __init__(self,opt, is_test=False):
        super(SGNGenerator, self).__init__()
        self.ngpu = len(opt.gpu_ids)

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(108, 64, 7, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),

        )

        self.res_block1 = SGNResidualBlock()
        self.res_block2 = SGNResidualBlock()
        self.res_block3 = SGNResidualBlock()
        self.res_block4 = SGNResidualBlock()
        self.res_block5 = SGNResidualBlock()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, padding=0),
            nn.Tanh()

        )

        self.apply(init_weights)

    def forward(self, z, seg, att):

        seg = seg.type(torch.cuda.FloatTensor)
        z_seg = torch.cat((z, seg), 1)
        if isinstance(z_seg.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            seg_feat = nn.parallel.data_parallel(self.encoder, z_seg, range(self.ngpu))
        else:
            seg_feat = self.encoder(z_seg)

        out1 = self.res_block1(seg_feat, att)
        out2 = self.res_block2(out1, att)
        out3 = self.res_block3(out2, att)
        out4 = self.res_block4(out3, att)
        out5 = self.res_block5(out4, att)

        if isinstance(out5.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.decoder, out5, range(self.ngpu))
        else:
            output = self.decoder(out5)

        return output



#Discriminator Class


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d,
                     use_sigmoid=False, num_D=3, getIntermFeat=False):
            super(MultiscaleDiscriminator, self).__init__()
            self.num_D = num_D
            self.n_layers = n_layers
            self.getIntermFeat = getIntermFeat

            for i in range(num_D):
                netD = SGNNLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
                if getIntermFeat:
                    for j in range(n_layers + 2):
                        setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
                else:
                    setattr(self, 'layer' + str(i), netD)

            self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
            self.downsampleSeg = nn.UpsamplingNearest2d(scale_factor=0.5)

    def singleD_forward(self, model, input, segmentation, attribute):
            if self.getIntermFeat:
                result = [input]
                for i in range(len(model)):
                    result.append(model[i](result[-1], attribute))
                return result[1:]
            else:
                return [model(input, segmentation, attribute)]

    def forward(self, input, segmentation, attribute):
            num_D = self.num_D
            result = []
            input_downsampled = input
            segmentation_downsampled = segmentation
            for i in range(num_D):
                if self.getIntermFeat:
                    model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                             range(self.n_layers + 2)]
                else:
                    model = getattr(self, 'layer' + str(num_D - 1 - i))
                result.append(self.singleD_forward(model, input_downsampled, segmentation_downsampled, attribute))
                if i != (num_D - 1):
                    input_downsampled = self.downsample(input_downsampled)
                    segmentation_downsampled = self.downsampleSeg(segmentation_downsampled)
            return result




#Perceptual Loss



#Gan Loss




#Model Builder
# returns generator and discriminator