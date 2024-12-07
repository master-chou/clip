import json
import os
import random
from collections import defaultdict
import alpha_clip
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from lvis import LVIS
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
from torchvision import transforms
from tqdm import tqdm
import pickle
import cv2
import torch
import numpy as np
import copy
from transformers import AutoProcessor
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
MASK_FILL = [int(255 * c) for c in PIXEL_MEAN]
def _convert_image_to_rgb(image):
    return image.convert("RGB")
clip_standard_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

hi_clip_standard_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((336, 336), interpolation=Image.BICUBIC),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

mask_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((224, 224)),
    transforms.Normalize(0.5, 0.26)
])

hi_mask_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((336, 336)),
    transforms.Normalize(0.5, 0.26)
])

def crop(image: np.array, bbox_xywh: np.array, bi_mask: np.array, scale=1.5):
    tl_x = int(bbox_xywh[0])
    tl_y = int(bbox_xywh[1])
    w = int(bbox_xywh[2]) if int(bbox_xywh[2]) > 0 else 1
    h = int(bbox_xywh[3]) if int(bbox_xywh[3]) > 0 else 1
    image_h, image_w = image.shape[:2]

    # shape maintained
    r = max(h, w)
    tl_x -= (r - w) / 2
    tl_y -= (r - h) / 2
    half_scale = (scale - 1.0) / 2
    w_l = int(tl_x - half_scale * r) if (tl_x - half_scale * r) > 0 else 0
    w_r = int(tl_x + (1+half_scale) * r) if (tl_x + (1+half_scale) * r) < image_w else image_w - 1
    h_t = int(tl_y - half_scale * r) if (tl_y - half_scale * r) > 0 else 0
    h_b = int(tl_y + (1+half_scale) * r) if (tl_y + (1+half_scale) * r) < image_h else image_h - 1

    return image[h_t: h_b, w_l: w_r, :], bi_mask[h_t: h_b, w_l: w_r]

def masked_crop(image: np.array, bbox_xywh: np.array, bi_mask: np.array, crop_scale=1.0, masked_color=[255, 255, 255]):
    # padding to make_sure bboxshape maintained
    image = np.pad(image, ((600, 600), (600, 600), (0, 0)), 'constant', constant_values=255)
    bi_mask = np.pad(bi_mask, ((600, 600), (600, 600)), "constant", constant_values=0)
    bbox_xywh[:2] += 600
    cropped_image, cropped_mask = crop(image, bbox_xywh, bi_mask, crop_scale)   
    # cropped_image[np.nonzero(cropped_mask == 0)] = MASK_FILL
    return cropped_image, cropped_mask

class COCO_Masked_Test(Dataset):
    def __init__(self, ann_file="data/coco/annotations/instances_val2017.json",  masked_color=[255, 255, 255], root_directory="data/coco/val2017", hi_res=False):
        self.masked_color = masked_color
        self.coco = COCO(annotation_file=ann_file)
        self.image_directory = root_directory
        self.crop_scale = 1.5
        self.anns_list = list(self.coco.anns.keys())
        self.index2id = [x['id'] for x in self.coco.cats.values()]
        self.id2index = dict()
        for i, item in enumerate(self.index2id):
            self.id2index[item] = i
        self.class_num = 80
        self.classes = [x['name'] for x in self.coco.cats.values()]
        
        if hi_res:
            self.mask_transform = hi_mask_transform
            self.clip_standard_transform = hi_clip_standard_transform
        else:
            self.mask_transform = mask_transform
            self.clip_standard_transform = clip_standard_transform
        
    def __len__(self):
        return len(self.anns_list)

    def __getitem__(self, index):
        ann_id = self.anns_list[index]
        ann = self.coco.anns[ann_id]
        img_id = self.coco.anns[ann_id]['image_id']
        image = np.array(Image.open(os.path.join(self.image_directory, self.coco.imgs[img_id]['file_name'])).convert('RGB'))
        bbox_xywh = np.copy(np.array(ann['bbox']))
        binary_mask = self.coco.annToMask(ann)
        cropped_image, cropped_mask =  masked_crop(image, bbox_xywh, binary_mask, crop_scale=self.crop_scale, masked_color=self.masked_color)
        image = self.clip_standard_transform(cropped_image)
        mask_torch = self.mask_transform(cropped_mask * 255)
        return image, mask_torch, self.id2index[ann['category_id']]

class LVIS_Masked_Test(Dataset):
    def __init__(self, ann_file="data/lvis/annotations/lvis_v1_val.json",  masked_color=[255, 255, 255], hi_res=False):
        self.masked_color = masked_color
        self.lvis = LVIS(ann_file)
        self.crop_scale = 1.5
        self.anns_list = list(self.lvis.anns.keys())
        self.index2id = [x['id'] for x in self.lvis.cats.values()]
        self.id2index = dict()
        for i, item in enumerate(self.index2id):
            self.id2index[item] = i
        self.class_num = 1203
        self.classes = [x['name'] for x in self.lvis.cats.values()]
        
        if hi_res:
            self.mask_transform = hi_mask_transform
            self.clip_standard_transform = hi_clip_standard_transform
        else:
            self.mask_transform = mask_transform
            self.clip_standard_transform = clip_standard_transform

    def __len__(self):
        return len(self.anns_list)

    def __getitem__(self, index):
        ann_id = self.anns_list[index]
        ann = self.lvis.anns[ann_id]
        img_id = self.lvis.anns[ann_id]['image_id']
        image = np.array(Image.open(self.lvis.imgs[img_id]['coco_url'].replace('http://images.cocodataset.org', 'data/coco')).convert('RGB'))
        binary_mask = self.lvis.ann_to_mask(ann)
        rgba = np.concatenate((image, np.expand_dims(binary_mask, axis=-1)), axis=-1)
        h, w = rgba.shape[:2]
        if max(h, w) == w:
            pad = (w - h) // 2
            l, r = pad, w - h - pad
            rgba = np.pad(rgba, ((l, r), (0, 0), (0, 0)), 'constant', constant_values=0)
        else:
            pad = (h - w) // 2
            l, r = pad, h - w - pad
            rgba = np.pad(rgba, ((0, 0), (l, r), (0, 0)), 'constant', constant_values=0)
        rgb = rgba[:, :, :-1]
        mask = rgba[:, :, -1]
        image = self.clip_standard_transform(rgb)
        mask_torch = self.mask_transform(mask * 255)
        return image, mask_torch, self.id2index[ann['category_id']], 

class RGBD:
    def __init__(self, annotation_file=None):
        self.anns, self.imgs, self.answers, self.types = defaultdict(list), dict(), dict(), dict()
        if not annotation_file == None:
            with open(annotation_file, 'r') as reader:
                datas = json.load(reader)
                for data in datas:
                    self.anns[data['id']] = data['captions']
                    self.imgs[data['id']] = data['image']
                    self.answers[data['id']] = data['answer']
                    self.types[data['id']] = data['type']

class RGBD_Outdoor_Benchmark(Dataset):
    def __init__(self, root_dir,tasks):
        self.root_dir = root_dir
        # import pdb;pdb.set_trace()
        self.dataset = RGBD(os.path.join(root_dir, tasks))
        self.image_ids = list(self.dataset.imgs.keys())
        self.captions = [x for x in self.dataset.anns.values()]
        # self.transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        # ])
        # self.transform =clip_standard_transform
        self.transform = hi_clip_standard_transform
        self.depth_transform = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_ids = self.image_ids[idx]
        image_path = os.path.join(self.root_dir, 'pic_all', self.dataset.imgs[img_ids])
        depth_path = os.path.join(self.root_dir, 'pic_depth' ,self.dataset.imgs[img_ids])
        image = Image.open(image_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')

        answer = self.dataset.answers[img_ids]

        if self.transform:
            image = self.transform(image)
        if self.depth_transform:
            depth = self.depth_transform(depth)
        return image, depth, answer


class RGBD_Benchmark_Test(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.dataset = RGBD(os.path.join(root_dir, 'annotations.json'))
        self.image_ids = list(self.dataset.imgs.keys())
        self.captions = [x for x in self.dataset.anns.values()]
        # self.transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        # ])
        self.transform =clip_standard_transform
        self.depth_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_ids = self.image_ids[idx]
        image_path = os.path.join(self.root_dir, 'all_pic', self.dataset.imgs[img_ids])
        depth_path = os.path.join(self.root_dir, 'depth-new' ,self.dataset.imgs[img_ids])
        image = Image.open(image_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')

        answer = self.dataset.answers[img_ids]

        if self.transform:
            image = self.transform(image)
        if self.depth_transform:
            depth = self.depth_transform(depth)
        return image, depth, answer

class RGBD_Benchmark_Test2(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.dataset = RGBD(os.path.join(root_dir, 'annotations2.json'))
        self.image_ids = list(self.dataset.imgs.keys())
        self.captions = [x for x in self.dataset.anns.values()]
        # self.transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        # ])
        self.transform =clip_standard_transform

        self.depth_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_ids = self.image_ids[idx]
        image_path = os.path.join(self.root_dir, 'all_pic', self.dataset.imgs[img_ids])
        depth_path = os.path.join(self.root_dir, 'depth-new' ,self.dataset.imgs[img_ids])
        image = Image.open(image_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')

        answer = self.dataset.answers[img_ids]

        if self.transform:
            image = self.transform(image)
        if self.depth_transform:
            depth = self.depth_transform(depth)
        return image, depth, answer
class ScanRefer:
    def __init__(self, annotation_file=None):
        self.anns, self.imgs, self.answers, self.scene_id = defaultdict(list), dict(), dict(), dict()
        if not annotation_file == None:
            with open(annotation_file, 'r') as reader:
                datas = json.load(reader)
                for data in datas:
                    self.anns[data['unique_id']] = data['descriptions']
                    self.imgs[data['unique_id']] = data['image']
                    self.answers[data['unique_id']] = data['answer']
                    self.scene_id[data['unique_id']] = data['scene_id']

class ScanRefer_Test(Dataset):
    def __init__(self, root_dir, model):
        self.root_dir = root_dir
        self.dataset = ScanRefer(os.path.join(root_dir, 'scanrefer_annotations_all.json'))
        # self.dataset = ScanRefer(root_dir)
        self.model = model
        self.image_ids = list(self.dataset.imgs.keys())
        # self.transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        # ])
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.depth_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_ids = self.image_ids[idx]
        image_path = os.path.join(self.root_dir, self.dataset.scene_id[img_ids], 'color', self.dataset.imgs[img_ids])
        depth_path = os.path.join(self.root_dir, self.dataset.scene_id[img_ids], 'depth', self.dataset.imgs[img_ids].split('.')[0] + '.png')

        image = Image.open(image_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')

        if self.transform:
            image = self.transform(image)
        if self.depth_transform:
            depth = self.depth_transform(depth)
        
        caption = self.dataset.anns[img_ids]
        texts = alpha_clip.tokenize(caption).cuda()
        text_embeddings = self.model.encode_text(texts)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        
        answer = self.dataset.answers[img_ids]
        return image, depth, text_embeddings, answer

class ScanRefer_Test2(Dataset):
    def __init__(self, root_dir, model):
        self.root_dir = root_dir
        self.dataset = ScanRefer(os.path.join(root_dir, 'annotations_2.json'))
        # self.dataset = ScanRefer(root_dir)
        self.model = model
        self.image_ids = list(self.dataset.imgs.keys())
        # self.transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        # ])
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.depth_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_ids = self.image_ids[idx]
        image_path = os.path.join(self.root_dir, self.dataset.scene_id[img_ids], 'color', self.dataset.imgs[img_ids])
        depth_path = os.path.join(self.root_dir, self.dataset.scene_id[img_ids], 'depth', self.dataset.imgs[img_ids].split('.')[0] + '.png')

        image = Image.open(image_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')

        if self.transform:
            image = self.transform(image)
        if self.depth_transform:
            depth = self.depth_transform(depth)
        
        caption = self.dataset.anns[img_ids]
        texts = alpha_clip.tokenize(caption).cuda()
        text_embeddings = self.model.encode_text(texts)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        
        answer = self.dataset.answers[img_ids]
        return image, depth, text_embeddings, answer

class ScanRefer_Testnr3d(Dataset):
    def __init__(self, root_dir, model):
        self.root_dir = root_dir
        self.dataset = ScanRefer(os.path.join(root_dir, 'nr3d_annotations.json'))
        # self.dataset = ScanRefer(root_dir)
        self.model = model
        self.image_ids = list(self.dataset.imgs.keys())
        # self.transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        # ])
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.depth_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_ids = self.image_ids[idx]
        image_path = os.path.join(self.root_dir, self.dataset.scene_id[img_ids], 'color', self.dataset.imgs[img_ids])
        depth_path = os.path.join(self.root_dir, self.dataset.scene_id[img_ids], 'depth', self.dataset.imgs[img_ids].split('.')[0] + '.png')

        image = Image.open(image_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')

        if self.transform:
            image = self.transform(image)
        if self.depth_transform:
            depth = self.depth_transform(depth)
        
        caption = self.dataset.anns[img_ids]
        texts = alpha_clip.tokenize(caption).cuda()
        text_embeddings = self.model.encode_text(texts)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        
        answer = self.dataset.answers[img_ids]
        return image, depth, text_embeddings, answer


if __name__ == "__main__":
    data = LVIS_Masked_Test()
    for i in tqdm(range(data.__len__())):
        data.__getitem__(i)
