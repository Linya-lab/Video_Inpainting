import os
import sys
import torch
import logging
import argparse
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

from tqdm import tqdm
from torch import optim
from torchvision import models
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from model.network import generator,discriminator
from dataset.dataset import videoInpaintData
from model.loss import AdversarialLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def eval(model,args,writer,epoch):
    model.eval()
    val_data=videoInpaintData(args.data_root,'valid',args.n_frames)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=4,drop_last=True)
    with torch.no_grad():
        for val_step,(frames,masks) in enumerate(val_loader):
            frames,masks=frames.cuda(),masks.cuda()
            b,t,c,h,w=frames.size()
            masked_frames=frames*(1.-masks)
            pred_imgs=model(masked_frames,masks)
            pred_comps=pred_imgs*masks+frames*(1-masks)
            masked_frames=masked_frames.masked_fill(masks==1,-1)
            writer.add_images('images/input',masked_frames.squeeze(0)*0.5+0.5,val_step+epoch/10-10)
            writer.add_images('images/output',pred_comps.squeeze(0)*0.5+0.5,val_step+epoch/10-10)
            writer.add_images('images/mask',masks.squeeze(0),val_step+epoch/10-10)
            if val_step==9:
                break

#训练前参数的设定
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',type=str, default='/data/dataset/YouTube_VOS',
                        help='Data root')     
    parser.add_argument('--checkpoint',type=str, default='/data/ppchu/video_inpainting/checkpoint',
                        help='Checkpoint root')
    parser.add_argument('--n_frames', type=int, default=5,
                        help='N_frames in each video')                              
    parser.add_argument('--epochs',type=int, default=1800,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('--batch_size', type=int,default=8,
                        help='Batch size')
    parser.add_argument('--lr',type=float,default=0.0001,
                        help='Learning rate')
    parser.add_argument('--load',type=bool,default=False,
                        help='Load trained model')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    net_G = generator()
    net_G= net_G.cuda()
    print(net_G)
    net_D = discriminator()
    net_D= net_D.cuda()
    print(net_D)

    logging.info(f'''Starting training:
            Epochs:          {args.epochs}
            Batch size:      {args.batch_size}
            Learning rate:   {args.lr}
            Load Model:      {args.load}
        ''')

    if args.load:
        checkpoint=torch.load(os.path.join(args.checkpoint,'checkpoint_200.pth'))
        net_G.load_state_dict(checkpoint['netG'])
        net_D.load_state_dict(checkpoint['netD'])
        start_epoch=checkpoint['epoch']
    else:
        start_epoch=0
    
    optimG=optim.Adam(net_G.parameters(),lr=args.lr,betas=(0.9,0.999))
    scheduler_G=optim.lr_scheduler.StepLR(optimG,step_size=600,gamma=0.1)
    optimD=optim.Adam(net_D.parameters(),lr=args.lr,betas=(0.9,0.999))
    scheduler_D=optim.lr_scheduler.StepLR(optimD,step_size=600,gamma=0.1)

    net_G = torch.nn.DataParallel(net_G, device_ids=[0,1])
    net_D = torch.nn.DataParallel(net_D, device_ids=[0,1])

    criterion = nn.L1Loss()
    adversarial_loss=AdversarialLoss().cuda()
    writer = SummaryWriter(comment=f'Video_Inpainting')
    train_data = videoInpaintData(args.data_root,'train',args.n_frames)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last=True)
    n_train = len(train_loader)
    for epoch in range(start_epoch+1,args.epochs+1):
        net_G.train()
        net_D.train()
        with tqdm(total=n_train,desc=f'Epoch {epoch}/{args.epochs}', unit='batch') as pbar:
            for global_step, (frames,masks) in enumerate(train_loader):
                frames,masks=frames.cuda(),masks.cuda()
                b,t,c,h,w=frames.size()
                masked_frames=frames*(1.-masks)
                pred_imgs=net_G(masked_frames,masks)
                pred_comps=pred_imgs*masks+frames*(1-masks)

                real_vid_feat = net_D(frames.permute(0,2,1,3,4))
                fake_vid_feat = net_D(pred_comps.permute(0,2,1,3,4).detach())
                dis_real_loss = adversarial_loss(real_vid_feat,True,True)
                dis_fake_loss = adversarial_loss(fake_vid_feat,False,True)
                loss_dis = (dis_real_loss + dis_fake_loss)/2
                writer.add_scalar('Loss/dis', loss_dis.item(), global_step+(epoch-1)*n_train)
                optimD.zero_grad()
                loss_dis.backward()
                optimD.step()

                gen_vid_feat = net_D(pred_comps.permute(0,2,1,3,4))
                loss_gan = 0.01*adversarial_loss(gen_vid_feat,True,False)
                writer.add_scalar('Loss/gan', loss_gan.item(), global_step+(epoch-1)*n_train)

                loss_hole=criterion(pred_imgs*masks,frames*masks)/torch.mean(masks)
                writer.add_scalar('Loss/hole', loss_hole.item(), global_step+(epoch-1)*n_train)

                loss_valid=criterion(pred_imgs*(1-masks),frames*(1-masks))/torch.mean(1-masks)
                writer.add_scalar('Loss/valid', loss_valid.item(), global_step+(epoch-1)*n_train)

                loss_gen=loss_hole+loss_valid+loss_gan
                optimG.zero_grad()
                loss_gen.backward()
                optimG.step()
                pbar.update(1)
            scheduler_G.step()
            scheduler_D.step()
            if epoch%100==0:
                eval(net_G,args,writer,epoch)
                state={'netG':net_G.module.state_dict(),'optimG':optimG.state_dict(),'epoch':epoch,
                'netD':net_D.module.state_dict(),'optimD':optimD.state_dict()}
                torch.save(state,os.path.join(args.checkpoint,'checkpoint_%03d.pth'%(epoch)))
                torch.save(net_G.module.state_dict(),os.path.join(args.checkpoint,'netG_%03d.pth'%(epoch)))
    writer.close()