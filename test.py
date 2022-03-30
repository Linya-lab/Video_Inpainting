import os
import cv2
from PIL import Image
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import math
import logging
import argparse
import torch
import importlib
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch import optim
from model.network import generator
# from ablation1.network import generator
from torchvision import models
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse

from dataset.utils import create_random_shape_with_random_motion
from dataset.utils import Stack, ToTorchFormatTensor

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# import lpips
# loss_fn = lpips.LPIPS(net='alex')
# loss_fn.cuda()

# def Computer_lpips(x,y):
#     n=x.size(0)
#     out=0
#     for i in range(n):
#         res = loss_fn(x[i], y[i])
#         out+=res.item()
#     return out/n

from scipy import linalg
from i3d import InceptionI3d
i3d_model = None

def init_i3d_model():
    global i3d_model
    if i3d_model is not None:
        return
    i3d_model_weight = '/data/ppchu/video_decaptioning/pre_model/rgb_imagenet.pt'
    i3d_model = InceptionI3d(400, in_channels=3, final_endpoint='Logits')
    i3d_model.load_state_dict(torch.load(i3d_model_weight))
    i3d_model.cuda()

def get_i3d_activations(batched_video, target_endpoint='Logits', flatten=True, grad_enabled=False):
    """
    Get features from i3d model and flatten them to 1d feature,
    valid target endpoints are defined in InceptionI3d.VALID_ENDPOINTS
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )
    """
    init_i3d_model()
    with torch.set_grad_enabled(grad_enabled):
        feat = i3d_model.extract_features(batched_video, target_endpoint)
    if flatten:
        feat = feat.view(feat.size(0), -1)

    return feat

def get_fid_score(real_activations, fake_activations):
    """
    Given two distribution of features, compute the FID score between them
    """
    m1 = np.mean(real_activations, axis=0)
    m2 = np.mean(fake_activations, axis=0)
    s1 = np.cov(real_activations, rowvar=False)
    s2 = np.cov(fake_activations, rowvar=False)
    return calculate_frechet_distance(m1, s1, m2, s2)


# code from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        logger.warning(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +  # NOQA
            np.trace(sigma2) - 2 * tr_covmean)

def evaluate_fid_score(video1,video2):
    with torch.no_grad():
        output_i3d_activations=get_i3d_activations(video1).cpu().numpy()
        real_i3d_activations=get_i3d_activations(video2).cpu().numpy()
        fid_score = get_fid_score(real_i3d_activations, output_i3d_activations)
    return fid_score

def MSE(x, y):
    return compare_mse(x, y)

def PSNR(ximg, yimg):
    return compare_psnr(ximg, yimg, data_range=1)

def SSIM(y, t, value_range=1):
    return compare_ssim(y,t,data_range=value_range,multichannel=True)

def Evaluate(files_gt, files_pred, methods=[PSNR,MSE, SSIM]):
    n,_,_,_=files_gt.shape
    for meth in methods:
        res = 0.
        for i in range(n):
            res += meth(files_gt[i], files_pred[i])
        res /= float(n)
    return res

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])

def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if not i in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref//2))
        end_idx = min(length, f + ref_length * (num_ref//2))
        for i in range(start_idx, end_idx+1, ref_length):
            if not i in neighbor_ids:
                if len(ref_index) > num_ref:
                #if len(ref_index) >= 5-len(neighbor_ids):
                    break
                ref_index.append(i)
    return ref_index

dir_test='/data/dataset/YouTube_VOS'
model_G_path='/data/ppchu/video_inpainting/checkpoint/netG_700.pth'
# model_G_path='/data/ppchu/video_inpainting/ablation1/checkpoint/netG1_500.pth'
ref_length = 10
neighbor_stride = 5
w, h = 432, 240
num_ref = -1

if __name__ == '__main__':
    net_G=generator().cuda()
    data=torch.load(model_G_path)
    net_G.load_state_dict(data)
    net_G=net_G.cuda() 
    net_G.eval()
    videos_path='/data/dataset/DAVIS/train/JPEGImages/480p'
    masks_path='/data/dataset/DAVIS/train/Annotations/480p'
    # videos_path='/data/dataset/YouTube_VOS/test_all_frames/JPEGImages'
    videos=os.listdir(videos_path)
    videos=sorted(videos)
    with tqdm(total=len(videos), desc=f'Video {len(videos)}', unit='video') as pbar:
        score=[]
        video=videos[64]
        print(video)
        # for video in videos:
            # if video=='7eb9424a53':
        video_name=os.path.join(videos_path,video)
        mask_name=os.path.join(masks_path,video)
        all_frames=[os.path.join(video_name, name) for name in sorted(os.listdir(video_name))]
        all_masks=[os.path.join(mask_name, name) for name in sorted(os.listdir(mask_name))]
        # all_masks=create_random_shape_with_random_motion(len(all_frames),240,432)
        nums=len(all_frames)
        frames=[]
        masks= np.empty((nums, h, w), dtype=np.float32)
        for i in range(nums):
            img=Image.open(all_frames[i]).convert('RGB')
            img=img.resize((432,240))
            frames.append(img)
            mask=Image.open(all_masks[i]).convert('RGB')
            mask=mask.resize((432,240))
            mask=np.asarray(mask)
            mask=np.mean(mask,axis=2,keepdims=False)
            mask=np.array(mask>0,dtype=np.uint8)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)))
            masks[i]=mask
        frames=_to_tensors(frames).cuda().unsqueeze(0)*2-1
        masks=torch.from_numpy(np.transpose(masks, (0, 1, 2)).copy()).float().cuda().unsqueeze(0).unsqueeze(2)
        masked_frames=frames.masked_fill(masks==1,-1)
        comp_frames = [None]*nums
        for f in range(0, nums, neighbor_stride):
            neighbor_ids = [i for i in range(max(0, f-neighbor_stride), min(nums, f+neighbor_stride+1))]
            ref_ids = get_ref_index(f,neighbor_ids, nums)
            len_temp = len(neighbor_ids) + len(ref_ids)
            selected_imgs = frames[:1, neighbor_ids+ref_ids, :, :, :]
            selected_masks = masks[:1, neighbor_ids+ref_ids, :, :, :]
            with torch.no_grad():
                pred_imgs =net_G(selected_imgs*(1-selected_masks),selected_masks)
                pred_imgs=pred_imgs*selected_masks+selected_imgs*(1-selected_masks)
                # pred_imgs= (pred_imgs+ 1) / 2
                # pred_imgs= pred_imgs.squeeze(0).cpu().permute(0, 2, 3, 1).numpy()
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    if comp_frames[idx] is None:
                        comp_frames[idx] = pred_imgs[:,i].unsqueeze(1)
                    else:
                        comp_frames[idx] =comp_frames[idx]*0.5+ pred_imgs[:,i].unsqueeze(1)*0.5
        comp_frames=torch.cat(comp_frames,dim=1)
        for j in range(nums):
            pred=comp_frames[:,j].squeeze(0)*0.5+0.5
            pred=pred.permute(1,2,0).cpu().numpy()*255
            pred=cv2.cvtColor(pred,cv2.COLOR_BGR2RGB)
            cv2.imwrite('/data/ppchu/video_inpainting/result/preds/p%03d.png'%(j+1),pred)
            input=masked_frames[:,j].squeeze(0)*0.5+0.5
            input=input.permute(1,2,0).cpu().numpy()*255
            input=cv2.cvtColor(input,cv2.COLOR_BGR2RGB)
            cv2.imwrite('/data/ppchu/video_inpainting/result/inputs/i%03d.png'%(j+1),input)
            true=frames[:,j].squeeze(0)*0.5+0.5
            true=true.permute(1,2,0).cpu().numpy()*255
            true=cv2.cvtColor(true,cv2.COLOR_BGR2RGB)
            cv2.imwrite('/data/ppchu/video_inpainting/result/trues/t%03d.png'%(j+1),true)
            mask=masks[:,j].squeeze(0)
            mask=mask.permute(1,2,0).cpu().numpy()*255
            mask=cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
            cv2.imwrite('/data/ppchu/video_inpainting/result/masks/m%03d.png'%(j+1),mask)

            # preds=(comp_frames*0.5+0.5).squeeze(0).cpu().permute(0,2,3,1).numpy()
            # trues=(frames*0.5+0.5).squeeze(0).cpu().permute(0,2,3,1).numpy()
            # res=evaluate_fid_score(comp_frames.permute(0,2,1,3,4),frames.permute(0,2,1,3,4).cuda())
            # score.append(res)
            # pbar.set_postfix(**{'VFID(video)': res})
            # pred_imgs=comp_frames.squeeze(0).cuda()
            # true_imgs=frames.squeeze(0).cuda()
            # res=Computer_lpips(pred_imgs,true_imgs)
            # score.append(res)
            # pbar.set_postfix(**{'LPIPS(video)': res})
            # res=Evaluate(preds,trues,[SSIM])                
            # if math.isinf(res):
            #     continue
            # score.append(res)
            # pbar.set_postfix(**{'SSIM(video)': res})
        #     pbar.update(1)
        # print(np.mean(score))