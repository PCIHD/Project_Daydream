'''
todo implement model architecture here , Discriminator , generator
'''
import torch
from torch import nn
from torch.autograd import Variable

#Generator class
from Layers import init_weights, SGNNLayerDiscriminator, SGNResidualBlock
from Sceneparser import ModelBuilder


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

        self.res_block1 = SGNResidualBlock(opt)
        self.res_block2 = SGNResidualBlock(opt)
        self.res_block3 = SGNResidualBlock(opt)
        self.res_block4 = SGNResidualBlock(opt)
        self.res_block5 = SGNResidualBlock(opt)

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
    def __init__(self, input_nc,opt, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d,
                     use_sigmoid=False, num_D=3, getIntermFeat=False):
            super(MultiscaleDiscriminator, self).__init__()
            self.num_D = num_D
            self.n_layers = n_layers
            self.getIntermFeat = getIntermFeat

            for i in range(num_D):
                netD = SGNNLayerDiscriminator(input_nc, opt ,ndf,n_layers, norm_layer, use_sigmoid, getIntermFeat)
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





#Perceptual loss

class PerceptualLoss(nn.Module):
    def __init__(self,args):
        super(PerceptualLoss,self).__init__()
        self.criterion = nn.L1Loss()
        builder = ModelBuilder()
        net_encoder = builder.build_encoder(weights=args.weights_encoder).cuda()
        self.net_encoder=net_encoder.eval()
        self.m = nn.AvgPool2d(3,stride=2,padding=1)
        for p in self.net_encoder.parameters():
            p.requires_grad = False

    def forward(self,real,fake):
        xc = Variable(real.data.clone(),volatile=True)
        f_fake = self.net_encoder(self.m(fake))
        f_real = self.net_encoder(self.m(xc))
        f_xc_c = Variable(f_real.data,requires_grad=False)
        loss = self.criterion(f_fake,f_xc_c)
        return loss



#gan loss
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)







#Model Builder
# returns generator and discriminator
def create_model(args):
    ngpu = len(args.gpu_ids)
    G = SGNGenerator(args)
    D = MultiscaleDiscriminator(input_nc=3,opt=args)
    return G,D
