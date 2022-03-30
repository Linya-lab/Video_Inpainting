import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import math
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.spectral_norm import spectral_norm as _spectral_norm

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1,bias=True,activation=nn.LeakyReLU(0.2, inplace=True)):
        super(Conv2dBlock,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias)
        self.activation=activation

    def forward(self,xs):
        xs=self.conv(xs)
        if self.activation is not None:
            xs=self.activation(xs)
        return xs

class Deconv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True,activation=nn.LeakyReLU(0.2, inplace=True)):
        super(Deconv2dBlock,self).__init__()
        self.conv=Conv2dBlock(in_channels,out_channels,kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias,activation=activation)
    def forward(self,xs):
        xs=F.interpolate(xs,scale_factor=2,mode='bilinear',recompute_scale_factor=True,align_corners=True)
        xs=self.conv(xs)
        return xs

class Atten_module(nn.Module):
    def forward(self,img_size,patch_size,_query,_key,_value,masks):
        b,t,c,h,w=img_size
        d_k=c//len(patch_size)
        output=[]
        for (height,width),query,key,value in zip(patch_size,
                                                    torch.chunk(_query, len(patch_size), dim=1),
                                                    torch.chunk(_key, len(patch_size), dim=1),
                                                    torch.chunk(_value, len(patch_size), dim=1)):
            out_h,out_w=h//height,w//width
            query=query.view(b,t,d_k,out_h,height,out_w,width)
            query=query.permute(0,1,3,5,2,4,6).contiguous().view(b,t*out_h*out_w,d_k*height*width)
            key=key.view(b,t,d_k,out_h,height,out_w,width)
            key=key.permute(0,1,3,5,2,4,6).contiguous().view(b,t*out_h*out_w,d_k*height*width)
            value=value.view(b,t,d_k,out_h,height,out_w,width)
            value=value.permute(0,1,3,5,2,4,6).contiguous().view(b,t*out_h*out_w,d_k*height*width)
            scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(query.size(-1))
            mm=masks.view(b,t,1,out_h,height,out_w,width)
            mm=mm.permute(0,1,3,5,2,4,6).contiguous().view(b,t*out_h*out_w,height*width)
            mm=(mm.mean(-1)>0.5).unsqueeze(1).repeat(1,t*out_h*out_w,1)
            scores.masked_fill(mm,-1e9)
            atten=F.softmax(scores, dim=-1)
            val=torch.matmul(atten,value)
            val=val.view(b,t,out_h,out_w,d_k,height,width)
            val=val.permute(0,1,4,2,5,3,6).contiguous().view(b*t,d_k,h,w)
            output.append(val)
        output=torch.cat(output,dim=1)
        return output
    
#Patches_Atten
class Patches_Atten1(nn.Module):
    def __init__(self,patch_size,channels):
        super(Patches_Atten1,self).__init__()
        self.patch_size=patch_size   
        self.query_embedding = nn.Conv2d(channels,channels, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(channels,channels, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(channels,channels, kernel_size=1, padding=0)
        self.attention=Atten_module()
        self.linear=Conv2dBlock(channels,channels,3,1,1)
        self.feedforward=nn.Sequential(Conv2dBlock(channels,channels,3,1,1),
                                    Conv2dBlock(channels,channels,3,1,2,2))

    def forward(self,x):
        xs,ms,b=x['xs'],x['ms'],x['b']
        bt,c,h,w=xs.size()
        masks=(F.interpolate(ms,size=(h,w),mode='bilinear',align_corners=True)>0).float()
        t=bt//b
        _query = self.query_embedding(xs)
        _key = self.key_embedding(xs)
        _value = self.value_embedding(xs)
        atten_res=self.attention([b,t,c,h,w],self.patch_size,_query,_key,_value,masks)
        xs=xs+self.linear(atten_res)
        xs=xs+self.feedforward(xs)
        return {'xs':xs,'ms':ms,'b':b}

#Patches_Atten2
class Patches_Atten2(nn.Module):
    def __init__(self,patch_size,channels):
        super(Patches_Atten2,self).__init__()
        self.patch_size=patch_size
        self.query_embedding = nn.Conv2d(channels,channels, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(channels,channels, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(channels,channels, kernel_size=1, padding=0)
        self.attention=Atten_module()
        self.linear=Conv2dBlock(channels,channels,3,1,1)

    def forward(self,xds,xs,ms,b):
        bt,c,h,w=xds.size()
        masks=(F.interpolate(ms,size=(h,w),mode='bilinear',align_corners=True)>0).float()
        t=bt//b
        _query = self.query_embedding(xds)
        _key = self.key_embedding(xs)
        _value = self.value_embedding(xs)
        atten_res=self.attention([b,t,c,h,w],self.patch_size,_query,_key,_value,masks)
        xs=xs+self.linear(atten_res)
        return xs

class generator(BaseNetwork):
    def __init__(self,init_weights=True):
        super(generator,self).__init__()
        cnum=64
        self.conv1=nn.Sequential(Conv2dBlock(3,cnum,3,1,1),
                                Conv2dBlock(cnum,cnum,3,2,1))
        self.conv2=nn.Sequential(Conv2dBlock(cnum,cnum*2,3,1,1),
                                Conv2dBlock(cnum*2,cnum*2,3,2,1))
        self.conv3=nn.Sequential(Conv2dBlock(cnum*2,cnum*4,3,1,1),
                                Conv2dBlock(cnum*4,cnum*4,3,2,1))

        self.patches_atten=nn.Sequential(Patches_Atten1([(30,54),(10,18),(5,9),(3,3)],cnum*4),
                                Patches_Atten1([(30,54),(10,18),(5,9),(3,3)],cnum*4),
                                Patches_Atten1([(30,54),(10,18),(5,9),(3,3)],cnum*4),
                                Patches_Atten1([(30,54),(10,18),(5,9),(3,3)],cnum*4),
                                Patches_Atten1([(30,54),(10,18),(5,9),(3,3)],cnum*4),
                                Patches_Atten1([(30,54),(10,18),(5,9),(3,3)],cnum*4))

        self.patches_atten3=Patches_Atten2([(15,27),(6,18)],cnum*4)
        self.upconv3=Deconv2dBlock(cnum*8,cnum*2,3,1,1)
        self.patches_atten2=Patches_Atten2([(20,36),(10,18)],cnum*2)
        self.upconv2=Deconv2dBlock(cnum*4,cnum,3,1,1)
        self.patches_atten1=Patches_Atten2([(60,108),(30,54)],cnum)
        self.upconv1=Deconv2dBlock(cnum*2,cnum,3,1,1)
        self.outconv=Conv2dBlock(cnum,3,3,1,1,activation=nn.Tanh())
        
        if init_weights:
            self.init_weights()

    def forward(self,imgs,masks):
        b,t,c,h,w=imgs.size()
        imgs=imgs.contiguous().view(b*t,c,h,w)
        masks=masks.contiguous().view(b*t,1,h,w)
        #encoder
        x1=self.conv1(imgs)
        x2=self.conv2(x1)
        x3=self.conv3(x2)
        xas=self.patches_atten({'xs':x3,'ms':masks,'b':b})['xs']
        xa4=self.patches_atten3(xas,x3,masks,b)
        xd3=self.upconv3(torch.cat([xas,xa4],dim=1)) 
        xa3=self.patches_atten2(xd3,x2,masks,b) 
        xd2=self.upconv2(torch.cat([xd3,xa3],dim=1))
        xa2=self.patches_atten1(xd2,x1,masks,b)
        xd1=self.upconv1(torch.cat([xd2,xa2],dim=1))
        out=self.outconv(xd1)
        out=out.contiguous().view(b,t,c,h,w)
        return out

class discriminator(BaseNetwork):
    def __init__(self,use_spectral_norm=True,init_weights=True):
        super(discriminator, self).__init__()
        cnum = 64
        self.conv= nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=3, out_channels=cnum, kernel_size=(3,5,5), stride=(1,2,2),
                                    padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv3d(cnum, cnum*2, kernel_size=(3,5,5), stride=(1,2,2),
                                    padding=(1,2,2), bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
      
            spectral_norm(nn.Conv3d(cnum * 2, cnum * 4, kernel_size=(3,5,5), stride=(1,2,2),
                                    padding=(1,2,2), bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv3d(cnum * 4, cnum * 4, kernel_size=(3,5,5), stride=(1,2,2),
                                    padding=(1,2,2), bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv3d(cnum * 4, cnum * 4, kernel_size=(3,5,5), stride=(1,2,2),
                                    padding=(1,2,2), bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(cnum * 4, cnum * 4, kernel_size=(3,5,5),
                      stride=(1,2,2), padding=(1,2,2)))

        if init_weights:
            self.init_weights()

    def forward(self,xs):
        out= self.conv(xs)
        return out

def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module