import json
import os
import torch
import torch.utils.data as data
from pathlib import Path
from random import randrange
import numpy as np
import h5py
import pickle


from .loader import VideoLoaderHDF5
from .loader import AudioFeatureLoader
from .spatial_transforms import get_spatial_transform,get_val_spatial_transforms
HDF5_DIR=''
PKL_DIR=''
PROJECT_DIR=''

def get_dataset(annotation_data,mode):
    video_names = []
    video_labels = []

    for key in annotation_data.keys():
        if annotation_data[key]['subset'] == mode:
            video_names.append(key)
            video_labels.append(annotation_data[key]['label'])
    return video_names,video_labels


class KSDataset(data.Dataset):
    def __init__(self,
                annotation_path=os.path.join(PROJECT_DIR,'dataset/KS_train_val.json'),
                mode='training',
                spatial_transform=None,
                video_loader = None,
                audio_drop=0.0,
                visual_drop=0.0
                ):
        
        self.video_dir = HDF5_DIR
        self.audio_dir = PKL_DIR

        self.audio_drop=audio_drop
        self.visual_drop=visual_drop

        self.dataset,self.idx_to_class,self.n_videos = self.__make_dataset(self.video_dir,annotation_path,mode)

        self.spatial_transform = spatial_transform

        self.loader = video_loader
        
    
    def __make_dataset(self,video_dir,annotation_path,subset):
        with open(annotation_path) as f:
            annotation_data = json.load(f)
        class_labels = annotation_data['labels']
        annotation_data = annotation_data['database']
        
        video_names , video_labels = get_dataset(annotation_data,subset)
        
        class_to_idx = {label : i for i,label in enumerate(class_labels)}
        idx_to_class = {i : label for i,label in enumerate(class_labels)}

        n_videos = len(video_names)

        dataset = []
        max_len = 0

        for i in range(n_videos):
            
            label = video_labels[i]
            label_id = class_to_idx[label]

            video_path = os.path.join(video_dir,video_names[i] + ".hdf5")
            audio_path = os.path.join(self.audio_dir,video_names[i] + ".pkl")
            if not os.path.exists(video_path) or not os.path.exists(audio_path):
                continue

            sample = {
                'video': video_names[i],
                'label': label_id,
            }

            dataset.append(sample)
        return dataset,idx_to_class,n_videos

    def add_mask_visual(self, image, ratio):
        patch_w = 10
        patch_l = 10
        w_num = int(224 / patch_w)
        l_num = int(224 / patch_l)
        total_num = w_num * l_num
        patch_num = int(total_num * ratio)
        # print(total_num, patch_num)
        patch_list = np.random.choice(total_num, patch_num, replace=False)
        for index in patch_list:
            patch_x = index % w_num * patch_w
            patch_y = int(index / w_num) * patch_l
            
            image[:, patch_x:patch_x+patch_w, patch_y:patch_y+patch_l] = 0.0

        return image

    def add_mask_audio(self, image, ratio):
        patch_w = 10
        patch_l = 10
        w_num = int(224 / patch_w)
        l_num = int(224 / patch_l)
        total_num = w_num * l_num
        patch_num = int(total_num * ratio)
        # print(total_num, patch_num)
        patch_list = np.random.choice(total_num, patch_num, replace=False)
        for index in patch_list:
            patch_x = index % w_num * patch_w
            patch_y = int(index / w_num) * patch_l
            
            image[patch_x:patch_x+patch_w, patch_y:patch_y+patch_l] = 0.0

        return image

    def __len__(self):
        return len(self.dataset)
    

    def __loading(self, path, video_name):

        clip=None
        try:
            clip = self.loader(path)
        except Exception as e:
            print("path {} has error".format(path))
        
        len_clip = len(clip)
        clip = [clip[0],clip[int((len_clip-1)/2)],clip[len_clip-1]]
        
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
            if self.visual_drop>0.0:
                clip=[self.add_mask_visual(img,self.visual_drop) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3) # c t h w
        return clip
   
    def __load_audio(self,audio_path):
        with open(audio_path,"rb") as f:
            audio = pickle.load(f)

        if self.audio_drop>0.0:
            audio=self.add_mask_audio(audio,self.audio_drop)
        return audio

    def __getitem__(self, index):

        video_name = self.dataset[index]['video']

        video_path = os.path.join(self.video_dir,video_name + ".hdf5")
        label = self.dataset[index]['label']

        clip = self.__loading(video_path, video_name)

        audio_path = os.path.join(self.audio_dir,video_name + ".pkl")
        
        audio = self.__load_audio(audio_path)

        return  audio,clip,label



class VisualDataset(data.Dataset):
    def __init__(self,
                annotation_path=os.path.join(PROJECT_DIR,'dataset/KS_train_val.json'),
                mode='training',
                spatial_transform=None,
                video_loader = None
                ):
        
        self.video_dir = HDF5_DIR
        self.audio_dir = PKL_DIR

        self.dataset,self.idx_to_class,self.n_videos = self.__make_dataset(self.video_dir,annotation_path,mode)

        self.spatial_transform = spatial_transform

        self.loader = video_loader
        
    
    def __make_dataset(self,video_dir,annotation_path,subset):
        with open(annotation_path) as f:
            annotation_data = json.load(f)
        class_labels = annotation_data['labels']
        annotation_data = annotation_data['database']
        
        video_names , video_labels = get_dataset(annotation_data,subset)
        
        class_to_idx = {label : i for i,label in enumerate(class_labels)}
        idx_to_class = {i : label for i,label in enumerate(class_labels)}

        n_videos = len(video_names)

        dataset = []
        max_len = 0

        for i in range(n_videos):
            
            label = video_labels[i]
            label_id = class_to_idx[label]

            video_path = os.path.join(video_dir,video_names[i] + ".hdf5")
            audio_path = os.path.join(self.audio_dir,video_names[i] + ".pkl")
            if not os.path.exists(video_path) or not os.path.exists(audio_path):
                continue

            sample = {
                'video': video_names[i],
                'label': label_id,
            }

            dataset.append(sample)
        return dataset,idx_to_class,n_videos



    def __len__(self):
        return len(self.dataset)
    

    def __loading(self, path, video_name):

        clip=None
        try:
            clip = self.loader(path)
        except Exception as e:
            print("path {} has error".format(path))
        
        len_clip = len(clip)
        clip = [clip[0],clip[int((len_clip-1)/2)],clip[len_clip-1]]
        
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        return clip
   
    def __load_audio(self,audio_path):
        with open(audio_path,"rb") as f:
            audio = pickle.load(f)

        return audio

    def __getitem__(self, index):

        video_name = self.dataset[index]['video']

        video_path = os.path.join(self.video_dir,video_name + ".hdf5")
        label = self.dataset[index]['label']

        clip = self.__loading(video_path, video_name)

        # audio_path = os.path.join(self.audio_dir,video_name + ".pkl")
        
        # audio = self.__load_audio(audio_path)

        return  clip,label


class AudioDataset(data.Dataset):
    def __init__(self,
                annotation_path=os.path.join(PROJECT_DIR,'dataset/KS_train_val.json'),
                mode='training',
                spatial_transform=None,
                video_loader = None
                ):
        
        self.video_dir = HDF5_DIR
        self.audio_dir = PKL_DIR

        self.dataset,self.idx_to_class,self.n_videos = self.__make_dataset(self.video_dir,annotation_path,mode)

        self.spatial_transform = spatial_transform

        self.loader = video_loader
        
    
    def __make_dataset(self,video_dir,annotation_path,subset):
        with open(annotation_path) as f:
            annotation_data = json.load(f)
        class_labels = annotation_data['labels']
        annotation_data = annotation_data['database']
        
        video_names , video_labels = get_dataset(annotation_data,subset)
        
        class_to_idx = {label : i for i,label in enumerate(class_labels)}
        idx_to_class = {i : label for i,label in enumerate(class_labels)}

        n_videos = len(video_names)

        dataset = []
        max_len = 0

        for i in range(n_videos):
            
            label = video_labels[i]
            label_id = class_to_idx[label]

            video_path = os.path.join(video_dir,video_names[i] + ".hdf5")
            audio_path = os.path.join(self.audio_dir,video_names[i] + ".pkl")
            if not os.path.exists(video_path) or not os.path.exists(audio_path):
                continue

            sample = {
                'video': video_names[i],
                'label': label_id,
            }

            dataset.append(sample)
        return dataset,idx_to_class,n_videos



    def __len__(self):
        return len(self.dataset)
    

    def __loading(self, path, video_name):

        clip=None
        try:
            clip = self.loader(path)
        except Exception as e:
            print("path {} has error".format(path))
        
        len_clip = len(clip)
        clip = [clip[0],clip[int((len_clip-1)/2)],clip[len_clip-1]]
        
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        return clip
   
    def __load_audio(self,audio_path):
        with open(audio_path,"rb") as f:
            audio = pickle.load(f)

        return audio

    def __getitem__(self, index):

        video_name = self.dataset[index]['video']

        # video_path = os.path.join(self.video_dir,video_name + ".hdf5")
        label = self.dataset[index]['label']

        # clip = self.__loading(video_path, video_name)

        audio_path = os.path.join(self.audio_dir,video_name + ".pkl")
        
        audio = self.__load_audio(audio_path)

        return  audio,label



if __name__=='__main__':

    import argparse
    parser=argparse.ArgumentParser()

    parser.add_argument("--decrease_epoch",type = int,default = 10)
    parser.add_argument('--sample_size',type = int,default = 112)
    parser.add_argument('--sample_t_stride',type = int,default = 1)
    parser.add_argument('--train_crop',
                        default='random',
                        type=str,
                        help=('Spatial cropping method in training. '
                              'random is uniform. '
                              'corner is selection from 4 corners and 1 center. '
                              '(random | corner | center)'))
    parser.add_argument('--value_scale',
                        default=1,
                        type=int,
                        help=
                        'If 1, range of inputs is [0-1]. If 255, range of inputs is [0-255].')
    parser.add_argument("--scale_h", type=int, default=128,
                        help="Scale image height to")
    parser.add_argument("--scale_w", type=int, default=171,
                        help="Scale image width to")
    parser.add_argument('--train_crop_min_scale',
                        default=0.25,
                        type=float,
                        help='Min scale for random cropping in training')
    parser.add_argument('--train_crop_min_ratio',
                        default=0.75,
                        type=float,
                        help='Min aspect ratio for random cropping in training')
    parser.add_argument('--no_hflip',
                        action='store_true',
                        help='If true holizontal flipping is not performed.')
    parser.add_argument('--colorjitter',
                        action='store_true',
                        help='If true colorjitter is performed.')
    parser.add_argument('--train_t_crop',
                        default='random',
                        type=str,
                        help=('Temporal cropping method in training. '
                              'random is uniform. '
                              '(random | center)'))

    args=parser.parse_args()

    spatial_transforms=get_spatial_transform(opt=args)

    dataset=KSDataset(video_loader=VideoLoaderHDF5(),spatial_transform=spatial_transforms)


