'''
todo main training loop
handels training image to image translation training loop


'''
import argparse
import os
import asyncio

import numpy as np
import torch
from torch.utils import data
import torch
import random
from dataloader import SGNDataset
from torch.autograd import Variable
from Model import create_model, PerceptualLoss, GANLoss
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--img_root', type=str, required=True,
                    help='root directory that contains images')
parser.add_argument('--save_filename', type=str, required=True,
                    help='Save file name')
parser.add_argument('--num_threads', type=int, default=4,
                    help='number of threads for fetching data (default: 4)')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='number of epochs (default: 100)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size (default: 32)')
parser.add_argument('--learning_rate', type=float, default=0.0002,
                    help='learning rate (dafault: 0.0002)')
parser.add_argument('--lr_decay', type=float, default=0.5,
                    help='learning rate decay (dafault: 0.5)')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='beta1 for Adam optimizer (dafault: 0.5)')
parser.add_argument('--resume_train', action='store_true',
                    help='continue training from the latest epoch')

parser.add_argument('--manualSeed', type=int,
                    help='manual seed')

parser.add_argument('--gpu_ids', type=str, default='0',
                    help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
# Scene Parsing Model related arguments
parser.add_argument('--scene_parsing_model_path', required=True,
                    help='folder to model path')
parser.add_argument('--suffix', default='_best.pth',
                    help="which snapshot to load")
parser.add_argument('--arch_encoder', default='resnet34_dilated8',
                    help="architecture of net_encoder")
parser.add_argument('--fc_dim', default=2048, type=int,
                    help='number of features between encoder and decoder')
parser.add_argument('--r', default=False, type=bool,
                    help='check resume')

args = parser.parse_args()
args.weights_encoder = os.path.join(args.scene_parsing_model_path, 'encoder' + args.suffix)
args.weights_decoder = os.path.join(args.scene_parsing_model_path, 'decoder' + args.suffix)


#check for gpu

if  torch.cuda.is_available():
    print('Gpu avialable')
    device='cuda'
elif torch.backends.mps.is_available():
    device="mps"
else:
    print('Gpu not avialable')
    device="cpu"

#randomize seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 100000)
#handeling for multiple gpus
gpu_ids = []
for str_id in args.gpu_ids.split(','):
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)
args.gpu_ids = gpu_ids
args.device = device
#set gradient calculations
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


#create input vector
def init_z_foreach_layout(category_map, batchsize):
    numofseg = 150

    ZT = torch.FloatTensor(batchsize, 100, 256, 256)
    ZT.fill_(0.0)
    ZT = ZT.to(device)

    for j in range(numofseg + 1):

        mask = category_map.eq(j)

        if (mask.any()):
            z = torch.rand(batchsize, 100, 1, 1).to(device)
            z.resize_(batchsize, 100, 1, 1).normal_(0, 1)
            z = z.expand(batchsize, 100, 256, 256)
            mask = mask.unsqueeze(1)
            mask = mask.type(torch.FloatTensor)
            ZT = ZT.add_(z * mask.to(device))

    del mask, z, category_map
    return ZT


async def log_images():
    print(
        'Epoch [%d/%d], Iter [%d/%d], D_real: %.4f, D_misSeg: %.4f, D_misAtt: %.4f, D_fake: %.4f, G_fake: %.4f, Percept: %.4f'
        % (epoch + 1, args.num_epochs, i + 1, len(train_loader), avg_D_real_loss / (i + 1),
           avg_D_real_m_loss / (i + 1), avg_D_real_m2_loss / (i + 1), avg_D_fake_loss / (i + 1),
           avg_G_fake_loss / (i + 1),
           avg_percept_loss / (i + 1)))
    save_image((fake.data + 1) * 0.5, './examples/%d_%d_fake.png' % (epoch + 1, i + 1))
    save_image((img_G.data + 1) * 0.5, './examples/%d_%d_real.png' % (epoch + 1, i + 1))
    torch.save(G.state_dict(), args.save_filename + "_G_latest")
    torch.save(D.state_dict(), args.save_filename + "_D_latest")


if __name__=='__main__':
    print(args)
    print('loading Dataset')
    train_data = SGNDataset(args)
    train_loader = data.DataLoader(train_data,batch_size=args.batch_size,
                                   shuffle=True,num_workers = args.num_threads)
    print('Connecting nodes , fabicrating network')
    if(not args.r):
        G, D = create_model(args)
    else:
        G,D = torch.load( args.save_filename + "_G_latest" ) , torch.save( args.save_filename + "_D_latest" )
    start_epoch = 0
    if(args.resume_train):
        rf = open('log.txt','r')
        log = rf.readline()
        log = log.split(' ')
        start_epoch = int(log[0])
        print('loading last trained step')
        pretrained_dict = torch.load(args.save_filename + "_G_latest")
        model_dict = G.state_dict()
        for k,v in pretrained_dict.items():
            if k in model_dict and v.size() == model_dict[k].size():
                model_dict[k] = v
            else:
                print('K',k)

        G.load_state_dict(model_dict)
        D.load_state_dict(torch.load(args.save_filename+"_D_latest"))
    criterionGan = GANLoss(use_lsgan=True,device=device)
    criterionFeat = torch.nn.L1Loss()
    criterionPercept =  PerceptualLoss(args)
    G.to(device)
    D.to(device)
    g_optimizer = torch.optim.Adam(G.parameters(),lr=args.learning_rate,betas = (args.momentum,0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999))
    if not os.path.isdir('./examples'):
        os.mkdir('./examples')
    if not os.path.isdir('./model'):
        os.mkdir('./model')
    loop = asyncio.get_event_loop()
    with torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./tensorboard_logs"),
            with_stack=True,record_shapes=True,
    ) as profiler:
        for epoch in range(start_epoch,args.num_epochs):

            avg_D_real_loss = 0
            avg_D_real_m_loss = 0
            avg_D_real_m2_loss = 0
            avg_D_fake_loss = 0
            avg_G_fake_loss = 0
            avg_percept_loss = 0
            for i ,(img,att,seg,cat,nnseg) in enumerate(train_loader):
                profiler.step()
                bs = img.size(0)
                rnd_batch_num = np.random.randint(len(train_data),size=bs)
                rnd_att_list = [train_data[i][1] for i in rnd_batch_num]
                rnd_att_np = np.asarray(rnd_att_list)
                rnd_att = torch.from_numpy(rnd_att_np).float()

                #convert images to tensors and send to gpu
                seg = seg.type(torch.FloatTensor)
                nnseg = nnseg.type(torch.FloatTensor)
                img = Variable(img.to(device))
                att = Variable(att.to(device))
                rnd_att = Variable(rnd_att.to(device))
                seg = Variable(seg.to(device))
                nnseg = Variable(nnseg.to(device))
                cat = Variable(cat.to(device))
                Z = init_z_foreach_layout(cat, bs)

                img_norm = img * 2 - 1
                img_G = img_norm

                requires_grad(G, False)
                requires_grad(D, True)
                D.zero_grad()

                #calculate loss for real image with segmask and attributes
                real_logit = D(img_norm,seg,att)
                real_loss = criterionGan(real_logit,True)
                avg_D_real_loss+=real_loss.data.item()
                real_loss.backward()

                #calculate loss for real image with mismatch segmask and attributes

                real_m_logit = D(img_norm,nnseg,att)
                real_m_loss = 0.25 * criterionGan(real_m_logit,False)
                avg_D_real_m_loss += real_m_loss.data.item()
                real_m_loss.backward()

                # real image with mismatching attribute and accurate segmask
                real_m2_logit = D(img_norm, seg, rnd_att)
                real_m2_loss = 0.25 * criterionGan(real_m2_logit, False)
                avg_D_real_m2_loss += real_m2_loss.data.item()
                real_m2_loss.backward()


                #now for the majedaar stuff generating image

                    #prepare nn for fake images

                fake = G(Z, seg, att)
                fake_logit = D(fake.detach(), seg, att)
                fake_loss = 0.5 * criterionGan(fake_logit, False)
                avg_D_fake_loss += fake_loss.data.item()
                fake_loss.backward()
    #prep to train generator
                d_optimizer.step()

                requires_grad(G, True)
                requires_grad(D, False)
                G.zero_grad()

                fake = G(Z, seg, att)
                fake_logit = D(fake, seg, att)
                fake_loss = criterionGan(fake_logit, True)
                # vgg_loss =10 * criterionVGG(img_G, fake)
                percept_loss = 10 * criterionPercept(img_G, fake)
                avg_G_fake_loss += fake_loss.data.item()
                # avg_vgg_loss += vgg_loss.data.item()
                avg_percept_loss += percept_loss.data.item()
                G_loss = fake_loss + percept_loss
                G_loss.backward()
                g_optimizer.step()

                if i % 10 == 0:
                    loop.run_until_complete(log_images())
                    log_file = open("log.txt", "w")
                    log_file.write(str(epoch) + " " + str(i))
                    log_file.close()
            if (epoch + 1) % 10 == 0:
                torch.save(G.state_dict(), args.save_filename + "_G_" + str(epoch))
                torch.save(D.state_dict(), args.save_filename + "_D_" + str(epoch))
                torch.save(G.state_dict(), args.save_filename + "_G_latest" )
                torch.save(D.state_dict(), args.save_filename + "_D_latest" )


            loop.close()




        #todo write training loop
    #todo figure out a way to download resnet model in required folder





