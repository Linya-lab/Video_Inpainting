import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import cv2
import torch
import imageio
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from dataset.utils import create_random_shape_with_random_motion
from dataset.utils import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip

class videoInpaintData(data.Dataset):
    def __init__(self,path,split,n_frames):
        self.path=path
        self.split=split
        self.n_frames=n_frames
        videos_path=os.path.join(path,split+'_all_frames/JPEGImages')
        videos_lst = os.listdir(videos_path)
        self.video_names=[os.path.join(videos_path, name) for name in videos_lst]
        self._to_tensors = transforms.Compose([Stack(),ToTorchFormatTensor(),])

    def __getitem__(self, index):
        #获取视频帧
        video_name=self.video_names[index]
        all_frames=[os.path.join(video_name, name) for name in sorted(os.listdir(video_name))]
        all_masks=create_random_shape_with_random_motion(len(all_frames),240,432)
        nums=len(all_frames)
        start=torch.randint(0,nums-self.n_frames+1,(1,)).item()
        frame_indices=[]
        for j in range(self.n_frames):
            frame_indices.append(start)
            start+=1
        frames,masks=[],[]
        for idx in frame_indices:
            img=Image.open(all_frames[idx]).convert('RGB')
            img=img.resize((432,240))
            frames.append(img)
            masks.append(all_masks[idx])
        if self.split == 'train':
            frames = GroupRandomHorizontalFlip()(frames)
        frame_tensors = self._to_tensors(frames)*2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        return frame_tensors,mask_tensors

    def __len__(self):
        return len(self.video_names)