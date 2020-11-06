'''
todo main training loop
handels training image to image translation training loop


'''
import argparse
import os
import torch
from .dataloader import SGNDataset
from torch import data
from .Model import create_model, PerceptualLoss, GANLoss
from .Sceneparser import

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


# Scene Parsing Model related arguments
parser.add_argument('--scene_parsing_model_path', required=True,
                    help='folder to model path')
parser.add_argument('--suffix', default='_best.pth',
                    help="which snapshot to load")
parser.add_argument('--arch_encoder', default='resnet34_dilated8',
                    help="architecture of net_encoder")
parser.add_argument('--fc_dim', default=2048, type=int,
                    help='number of features between encoder and decoder')

args = parser.parse_args()
args.weights_encoder = os.path.join(args.scene_parsing_model_path, 'encoder' + args.suffix)
args.weights_decoder = os.path.join(args.scene_parsing_model_path, 'decoder' + args.suffix)
if not torch.cuda.is_avialable():
    print('No Gpu avialable')
    device='cpu'
else:
    device="cuda"


if __name__=='__main':
    print('loading Dataset')
    train_data = SGNDataset(args)
    train_loader = data.DataLoader(train_data,batch_size=args.batch_size,
                                   shuffle=True,num_workers = args.num_threads)
    print('Connecting nodes , fabicrating network')
    G, D = create_model(args)
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
    criterionGan = GANLoss(use_lsgan=True)
    criterionFeat = torch.nn.L1Loss()
    criterionPercept =  PerceptualLoss(args.gpu_ids,args)


