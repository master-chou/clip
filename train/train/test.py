import torch
# import alpha_clip
import alpha_clip_test as alpha_clip
# import alpha_clip_latefuse as alpha_clip
import os
import sys
import wandb
# 获取上级目录的绝对路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将 'train' 目录添加到 sys.path 中
sys.path.append(parent_dir)
# import pdb;pdb.set_trace()
from utils import concat_all_gather, is_dist_avail_and_initialized, accuracy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
# from dataset.imagenet_s_test import Imagenet_S
from dataset.mask_image_test import COCO_Masked_Test
# from dataset.alpha_grit import Alpha_GRIT
# from dataset.mask_image import ImageNet_Masked
from torch.utils.data.distributed import DistributedSampler
from dataset.mask_image_test import ScanRefer_Test,ScanRefer_Test2,ScanRefer_Testnr3d
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
from collections import defaultdict
import sys
sys.path.append('/home/aiops/wangzh/llava')
from depth_anything_v2.dpt import DepthAnythingV2
from scheduler import cosine_lr
import argparse
import os
import subprocess
import collections
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import loralib as lora
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import json
from dataset.mask_image_test import RGBD_Benchmark_Test,RGBD_Benchmark_Test2,RGBD_Outdoor_Benchmark
from torchvision.datasets import CocoCaptions
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from imagenet import imagenet_templates,imagenet_data_loader,get_label
simple_templates = [
    'a photo of a {}.'
]
hi_clip_standard_transform = Compose([
    ToTensor(), 
    Resize((336, 336), interpolation=Image.BICUBIC),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
clip_standard_transform = Compose([
    ToTensor(), 
    Resize((224, 224), interpolation=Image.BICUBIC),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
class BLINK:
    def __init__(self, annotation_file=None):
        self.anns, self.imgs, self.answers, self.task = defaultdict(list), dict(), dict(), dict()
        if not annotation_file == None:
            with open(annotation_file, 'r') as reader:
                datas = json.load(reader)
                for data in datas:
                    self.anns[data['idx']] = data['captions']
                    self.imgs[data['idx']] = data['image_1']
                    self.answers[data['idx']] = data['answer']
                    self.task[data['idx']] = data['sub_task']

class BLINK_Benchmark(Dataset):
    def __init__(self,tasks):
        self.root_dir = '/home/aiops/wangzh/data/blink/BLINK/Spatial_Relation'
        # self.root_dir = '/home/aiops/wangzh/data/blink/BLINK/Relative_Depth'
        print(self.root_dir)
        # import pdb;pdb.set_trace()
        self.dataset = BLINK(os.path.join(self.root_dir,'test.json'))
        self.image_ids = list(self.dataset.imgs.keys())
        # self.scene_ids = list(self.dataset.scene_id.keys())
        self.captions = [x for x in self.dataset.anns.values()]
        # self.transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        # ])
        self.transform =clip_standard_transform
        # self.transform = hi_clip_standard_transform
        self.depth_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = 'vitb'
    
        self.depth_model=DepthAnythingV2(**model_configs[encoder])
        self.depth_model.load_state_dict(torch.load(f'/home/aiops/wangzh/zss/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.depth_model.to(torch.float32)
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_ids = self.image_ids[idx]
        # scene = self.scene_ids[idx]
        image_path = os.path.join(self.root_dir, 'output_images',self.dataset.imgs[img_ids])
        # depth_path = os.path.join(self.root_dir, 'pic_depth' ,self.dataset.imgs[img_ids])
        image = Image.open(image_path).convert('RGB')
        
        # depth = Image.open(depth_path).convert('L')

        answer = self.dataset.answers[img_ids]

        if self.transform:
            image = self.transform(image)
        # if self.depth_transform:
        #     depth = self.depth_transform(depth)
            depth = self.depth_model(image.unsqueeze(0))
            depth = depth.squeeze(0)
        return image, depth, answer


class RGBD_INDOOR:
    def __init__(self, annotation_file=None):
        self.anns, self.imgs, self.answers, self.scene_id = defaultdict(list), dict(), dict(), dict()
        if not annotation_file == None:
            with open(annotation_file, 'r') as reader:
                datas = json.load(reader)
                for data in datas:
                    self.anns[data['id']] = data['captions']
                    self.imgs[data['id']] = data['image']
                    self.answers[data['id']] = data['answer']
                    self.scene_id[data['id']] = data['scene_id']

class RGBD_Indoor_Benchmark(Dataset):
    def __init__(self,tasks):
        self.root_dir = '/home/aiops/wangzh/data/scanner'
        # import pdb;pdb.set_trace()
        self.dataset = RGBD_INDOOR(os.path.join(self.root_dir,'indoor-new',tasks))
        self.image_ids = list(self.dataset.imgs.keys())
        # self.scene_ids = list(self.dataset.scene_id.keys())
        self.captions = [x for x in self.dataset.anns.values()]
        # self.transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        # ])
        # self.transform =clip_standard_transform
        self.transform = hi_clip_standard_transform
        # self.depth_transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        # ])
        self.depth_transform = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.ToTensor(),
        ])
        model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = 'vitb'
    
        self.depth_model=DepthAnythingV2(**model_configs[encoder])
        self.depth_model.load_state_dict(torch.load(f'/home/aiops/wangzh/zss/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.depth_model.to(torch.float32)
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_ids = self.image_ids[idx]
        # scene = self.scene_ids[idx]
        image_path = os.path.join(self.root_dir, 'scannet_2d_HR3', self.dataset.scene_id[img_ids],'color',self.dataset.imgs[img_ids])
        # depth_path = os.path.join(self.root_dir, 'pic_depth' ,self.dataset.imgs[img_ids])
        image = Image.open(image_path).convert('RGB')
        
        # depth = Image.open(depth_path).convert('L')

        answer = self.dataset.answers[img_ids]

        if self.transform:
            image = self.transform(image)
        # if self.depth_transform:
        #     depth = self.depth_transform(depth)
            depth = self.depth_model(image.unsqueeze(0))
            depth = depth.squeeze(0)
        return image, depth, answer


class DepthSpatialDataset(Dataset):
    def __init__(self, root_dir='', transform=None, depth_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.depth_transform = depth_transform
        
        json_file_path='/home/aiops/wangzh/data/spatial/filtered_data_all.json'
        with open(json_file_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        # 生成索引
        # self.image_filenames = [record['image_id'] for record in self.metadata]
        # import pdb;pdb.set_trace()
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 从 metadata 中获取图像和深度图文件名
        record = self.metadata[idx]
        img_name = record['image_id']
        spatial_caption = record['caption'].get('spatial_caption')
        image_path=f'/dataset/SA-1B/data/sa_{record["image_id"][3:-8]:0>6}/{record["image_id"]}'
        # image_path = os.path.join(self.root_dir, img_name + '.jpg')
        depth_path = f'/home/aiops/wangzh/data/depth-image/{record["image_id"]}'
        
        # 初始化默认返回值
        image, depth = None, None
        
        # 尝试加载图片
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}. Error: {e}")
        
        # 尝试加载深度图
        try:
            depth = Image.open(depth_path).convert('L')
            if self.depth_transform:
                depth = self.depth_transform(depth)
        except Exception as e:
            print(f"Warning: Failed to load depth image {depth_path}. Error: {e}")
        
        # 如果图像或深度图加载失败，则跳过该数据点
        if image is None or depth is None:
            return self.__getitem__((idx + 1) % len(self))  # 返回下一个有效的数据点
        
        return image, depth, spatial_caption


class CocoDataset(Dataset):
    def __init__(self, root, annFile, transform=None,depth_transform=None):
        self.coco = CocoCaptions(root=root, annFile=annFile, transform=None)
        self.transform = transform
        self.depth_transform = depth_transform
        self.image_paths=[]
        for i in range(len(self.coco)):
            img_info = self.coco.coco.loadImgs(self.coco.ids[i])[0]
            img_path = os.path.join(self.coco.root, img_info['file_name'])
            self.image_paths.append(img_path)
        # import pdb;pdb.set_trace()
    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        image, captions = self.coco[idx]
        img_path = self.image_paths[idx] 
        if self.transform:
            image = self.transform(image)
        depth_map_path = img_path.replace('val2017', 'depth2017')
        depth_map = Image.open(depth_map_path)
        if self.depth_transform:
            depth_map = self.depth_transform(depth_map)
        captions = captions[:5]  # 取前5个caption
        captions = clip.tokenize(captions)
        # print(f"Image shape: {image.shape}, Depth shape: {depth_map.shape}, Captions: {captions.shape}")
        # print(depth_map_path)
        return image, depth_map,captions



class DepthDataset(Dataset):
    def __init__(self, root_dir='', transform=None, depth_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.depth_transform = depth_transform
        
        # 读取 JSONL 文件中的 metadata
        self.metadata = []
        jsonl_file = '/home/aiops/wangzh/data/metadata_new.jsonl'
        with open(jsonl_file, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                self.metadata.append(record)
        # import pdb;pdb.set_trace()
        # 生成索引
        self.image_filenames = [record['image_id'] for record in self.metadata]
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 从 metadata 中获取图像和深度图文件名
        record = self.metadata[idx]
        img_name = record['image_id']
        caption = record.get('caption', '')  # 使用 get() 方法防止键不存在导致的异常
        image_path=f'/dataset/SA-1B/data/sa_{record["image_id"][3:-8]:0>6}/{record["image_id"]}'
        # image_path = os.path.join(self.root_dir, img_name + '.jpg')
        depth_path = f'/home/aiops/wangzh/data/depth-image/{record["image_id"]}'
        
        # 初始化默认返回值
        image, depth = None, None
        
        # 尝试加载图片
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}. Error: {e}")
        
        # 尝试加载深度图
        try:
            depth = Image.open(depth_path).convert('L')
            if self.depth_transform:
                depth = self.depth_transform(depth)
        except Exception as e:
            print(f"Warning: Failed to load depth image {depth_path}. Error: {e}")
        
        # 如果图像或深度图加载失败，则跳过该数据点
        if image is None or depth is None:
            return self.__getitem__((idx + 1) % len(self))  # 返回下一个有效的数据点
        
        return image, depth, caption

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 定义图像和深度图的变换
def _convert_image_to_rgb(image):
    return image.convert("RGB")
# image_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

depth_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
    
clip_transform = transforms.Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
def zeroshot_classifier_ours(captions, model, local_rank=0):
    with torch.no_grad():
        zeroshot_weights = []
        for caption in tqdm(captions, disable=(dist.get_rank() != 0)):
            texts = alpha_clip.tokenize(caption).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # class_embedding = class_embeddings.mean(dim=0)
            # class_embedding /= class_embedding.norm()
            # import pdb;pdb.set_trace()
            zeroshot_weights.append(class_embeddings)
        # zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()
    return zeroshot_weights

def zeroshot_classifier(classnames, templates, model, local_rank=0):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, disable=(dist.get_rank() != 0)):
            texts = [template.format(classname) for template in templates] #format with class
            texts = alpha_clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


    
def get_accuracy_i2t(text_feature, image_feature, k):
    with torch.no_grad():
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)

        text_feature = text_feature.cuda()
        image_feature = image_feature.cuda()

        pred_true = 0

        sim = (image_feature @ text_feature.T).softmax(dim=-1)
        for i in range(image_feature.shape[0]):
            pred = sim[i]
            values, topk = pred.topk(k)
            for j in range(5):
                true_index = 5*i + j
                if true_index in topk:
                    pred_true = pred_true + 1
                    break

        print(pred_true/image_feature.shape[0])


def get_accuracy_t2i(text_feature, image_feature, k):
    with torch.no_grad():
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)

        text_feature = text_feature.cuda()
        image_feature = image_feature.cuda()

        pred_true = 0

        sim = (text_feature @ image_feature.T).softmax(dim=-1)

        for i in range(text_feature.shape[0]):
            pred = sim[i]
            values, topk = pred.topk(k)
            true_index = i//5
            if true_index in topk:
                pred_true = pred_true + 1

        print(pred_true/text_feature.shape[0])



class CLIP_Clean_Train():
    def __init__(self, local_rank=0, lr=4e-5, weigth_decay=0.02, log_scale=4.6052, lora_rank=-1, common_pair=0.0, para_gamma=0.01, exp_name="auto", warmup_length=200, epoch_num=1, subnum=10000,pos_embed=None):
        self.local_rank = local_rank
        self.pos_embed=pos_embed
        if lora_rank == -1:
            # self.model, _ = alpha_clip.load("ViT-L/14@336px", device='cpu', lora_adapt=False, rank=-1)
            # self.model, _ = alpha_clip.load("ViT-L/14", device='cpu', lora_adapt=False, rank=-1)
            self.model, _ = alpha_clip.load("ViT-B/16", device='cpu', lora_adapt=False, rank=-1)
        else:
            # self.model, _ = alpha_clip.load("ViT-L/14", device='cpu', lora_adapt=True, rank=lora_rank)
            self.model, _ = alpha_clip.load("ViT-B/16", device='cpu', lora_adapt=True, rank=lora_rank)
        torch.cuda.set_device(device=f'cuda:{local_rank}')
        self.model = self.model.float().cuda()
        self.batch_size = 256
        self.num_epoch = int(1e10)
        self.lr = lr
        self.subnum = subnum
        if exp_name == "auto":
            self.logdir = f"log/grit_1m/lr={lr}_wd={weigth_decay}_wl={warmup_length}_logs={log_scale}_L14_336_lora={lora_rank}_cp={common_pair}_para_gamma={para_gamma}_e{self.num_epoch}_16xb_subnum={self.subnum}"
        else:
            self.logdir = exp_name
        self.ckptdir = self.logdir + "/ckpt/"
        os.makedirs(self.ckptdir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir)
        self.model.visual = torch.nn.parallel.DistributedDataParallel(self.model.visual, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        # logit scale
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * log_scale)
        conv_opt_paras = []
        other_opt_paras = []
        if lora_rank != -1: # use lora
            lora.mark_only_lora_as_trainable(self.model)
            for k, v in self.model.named_parameters():
                if v.requires_grad:
                    other_opt_paras.append(v)
                elif "conv1_alpha" in k:
                    v.requires_grad_(True)
                    conv_opt_paras.append(v)
        else: # normal to not use lora
            for k, v in self.model.named_parameters():
                v.requires_grad_(False)
            for k, v in self.model.visual.named_parameters():
                v.requires_grad_(True)
                if "conv1_alpha" in k:
                    conv_opt_paras.append(v)
                else:
                    other_opt_paras.append(v) 
        self.optimizer = optim.AdamW(
            [
                {"params": conv_opt_paras, "lr": self.lr},
                {"params": other_opt_paras, "lr": self.lr * para_gamma}
            ],
        )
        self.para_gamma = para_gamma
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()
        self.model.visual.requires_grad_(True)
        checkpoint=os.listdir(self.ckptdir)
        checkpoint.sort(key=lambda x: int(x[5:-4]))

        # self.model, preprocess = clip.load("ViT-B/16", device='cuda')

        resume_pth ="iter_8000.pth"
        map_location = {'cuda:0': f'cuda:{self.local_rank}'}
        self.model.load_state_dict(torch.load(os.path.join(self.ckptdir, resume_pth), map_location=map_location),strict=False)
        print(f"load resumed checkpoint:{self.ckptdir} {resume_pth}")  
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params}")
        
        # import pdb;pdb.set_trace()
        # self.print_trainable_params()



    def print_trainable_params(self):
        print("Trainable parameters:")
        for name, param in self.model.named_parameters():
            # if param.requires_grad:
            print(name)

    def inference(self, images, masks, texts):
        # import pdb;pdb.set_trace()
        image_features = self.model.visual(images, masks,pos_embed=self.pos_embed)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.model.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        image_feat_all = concat_all_gather(image_features)
        text_feat_all = concat_all_gather(text_features)
        
        sim_i2t = torch.matmul(image_features, text_feat_all.T)
        sim_t2i = torch.matmul(image_feat_all, text_features.T)
        sim_t2i = sim_t2i.T

        sim_i2t = self.model.logit_scale.exp() * sim_i2t
        sim_t2i = self.model.logit_scale.exp() * sim_t2i

        if is_dist_avail_and_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        bs = images.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            images.device
        )
        loss_itc = (
                F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
            ) / 2
        return loss_itc

    def train_epoch(self, dataloader, test_loaders, epoch, start_iter=0, amp=False):
        running_loss = 0.0
        num_batches_per_epoch = len(dataloader)
        for i, (images, masks, texts) in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
            # import pdb;pdb.set_trace()
            step = num_batches_per_epoch * epoch + i
            if step < start_iter:
                continue
            self.optimizer.zero_grad()
            self.scheduler(step)
            images = images.cuda()
            masks = masks.cuda()
            texts = alpha_clip.tokenize(texts).cuda()
            if amp:
                with torch.cuda.amp.autocast():
                    loss = self.inference(images, masks, texts)
                    
                # import pdb;pdb.set_trace()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.inference(images, masks, texts)
                loss.backward()
                self.optimizer.step()
            running_loss += loss.item()
            batch_num = i + 1
        
            loss = running_loss / 400
            running_loss = 0.0
            loss = torch.tensor(loss).cuda()
            dist.all_reduce(loss)
            loss = loss.item() / torch.distributed.get_world_size()
            if step % 200 == 0:
                if dist.get_rank() == 0:

                    print("=====================================")
                    print(f"train lr (alpha conv) step {step}: {self.optimizer.param_groups[0]['lr']}")
                    print(f"train lr (other layer) step {step}: {self.optimizer.param_groups[1]['lr']}")
                    print(f"train logit_scale step {step}: {self.model.logit_scale.item()}")
                    print(f"train loss step {step}: {loss}")
                    print("=====================================")

                    if step % 10000 == 0 and step != 0 and step > 800:
                        torch.save(self.model.visual.state_dict(), self.ckptdir + f'iter_{step}.pth')
                    
                with torch.no_grad():
                    self.model.visual.eval()
                    for test_name, test_loader in test_loaders.items():
                        # self.text_embeddings = zeroshot_classifier(test_loader.dataset.classes, simple_templates, self.model, self.local_rank)
                        if test_name == 'OURS':
                            self.text_embeddings = zeroshot_classifier_ours(test_loader.dataset.captions, self.model, self.local_rank)
                            acc1 = self.test_epoch_ours(test_loader)
                            print("=====================================")
                            print(f"RGBD-bench test mean of per class acc-1 step 0: {acc1}")
                            print("=====================================")
                            

                        if test_name == 'COCO':
                            # self.text_embeddings = zeroshot_classifier(test_loader.dataset.classes, simple_templates, self.model, self.local_rank)
                            i2t_accuracies, t2i_accuracies = self.test_epoch_retrieval(test_loader)
                            print("=====================================")
                            print(f"CoCo test i2t_accuracies: {i2t_accuracies}")
                            print(f"CoCo test t2i_accuracies: {t2i_accuracies}")
                            print("=====================================")
                            
                        
                            
                    self.model.visual.train()
        return running_loss / batch_num
    def zeroshot_classifier_imagenet(self,model, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                # import pdb;pdb.set_trace()
                texts = [template.format(classname) for template in templates]  # format with class
                texts = clip.tokenize(texts).cuda()  # tokenize
                class_embeddings = model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    @torch.no_grad()
    def test_imagenet(self):
        self.model.eval()
        imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]
        modelss, preprocess = clip.load("ViT-B/16")
        loader, dataset = imagenet_data_loader(preprocess)
        # import pdb;pdb.set_trace()
        zeroshot_weights = self.zeroshot_classifier_imagenet(self.model, imagenet_classes, imagenet_templates)
        total_num = 0
        true_num = 0
        model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = 'vitb'
        softmax = torch.nn.Softmax(dim=1)
        depth_model=DepthAnythingV2(**model_configs[encoder])
        depth_model.load_state_dict(torch.load(f'/home/aiops/wangzh/zss/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        depth_model.to(torch.float32).to('cuda')
        with torch.no_grad():
            for i, (images, targets, paths) in enumerate(tqdm(loader)):
                images = images.to('cuda')
                # import pdb;pdb.set_trace()
                depth = None
                depth = depth_model(images)

                # image_features = self.model.encode_image(images,depth)
                image_features = self.model.visual(images,depth,pos_embed=self.pos_embed)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100. * image_features @ zeroshot_weights
                logits = softmax(logits)
                pred = torch.argmax(logits,dim=1)
                
                total_len = pred.shape[0]
                for i in range(total_len):
                    label = targets[i]
                    label = get_label(label).item()
                    if pred[i].item() == label:
                        true_num += 1
                    total_num += 1
            
            #save_to_file(logits, targets, paths)
        print(true_num / total_num)

    @torch.no_grad()
    def test_epoch_retrieval(self, dataloader):
        image_features = []
        text_features = []
        coco_dataset = CocoDataset(root="/home/aiops/wangzh/data/val2017", annFile="/home/aiops/wangzh/data/captions_val2017.json", transform=clip_transform,depth_transform=depth_transform)
        dataloader = torch.utils.data.DataLoader(coco_dataset, batch_size=256, num_workers=4, pin_memory=True)


        with torch.no_grad():
            for images, depth,captions_batch in dataloader:
                images = images.to('cuda')
                # import pdb;pdb.set_trace()
                image_features.append(self.model.visual(images,depth,pos_embed=self.pos_embed))
                # captions_batch = list(map(list, zip(*captions_batch)))
                # import pdb;pdb.set_trace()
                for captions in captions_batch:
                    # caption_input = alpha_clip.tokenize(captions).cuda()
                    caption_input=captions.cuda()
                    text_features.append(self.model.encode_text(caption_input))
            
            # import pdb;pdb.set_trace()
            image_features = torch.cat(image_features) #[5000,512]

            image_features /= image_features.norm(dim=-1, keepdim=True)

            text_features = torch.stack(text_features) #[5000,5,512]
            text_features = text_features.view(-1,text_features.size(-1))
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = image_features @ text_features.T
            # I2T (Image to Text) 检索
            i2t_accuracies = []
            for k in [1, 5, 10]:
                pred_true = 0
                for i in range(5000):
                    pred = similarity[i]
                    b = pred.argsort()[-k:]
                    for j in range(5):
                        true_index = 5 * i + j
                        if true_index in b:
                            pred_true += 1
                            break
                i2t_accuracies.append(pred_true / 5000)
                print("acc",pred_true / 5000)

            # T2I (Text to Image) 检索
            t2i_accuracies = []
            similarity = similarity.T
            for k in [1, 5, 10]:
                pred_true = 0
                for i in range(25000):
                    pred = similarity[i]
                    b = pred.argsort()[-k:]
                    true_index = i // 5
                    if true_index in b:
                        pred_true += 1
                t2i_accuracies.append(pred_true / 25000)
                print("acc",pred_true / 25000)

        return i2t_accuracies, t2i_accuracies
    @torch.no_grad()
    def test_epoch_blink(self, dataloader):
        corr_pred = 0
        total_num = 0
        for i, (images, depth, answer) in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
            # import pdb; pdb.set_trace()
            images = images.cuda()
            answer = answer.cuda()
            # import pdb;pdb.set_trace()
            image_features = self.model.visual(images, depth,pos_embed=self.pos_embed)
            # image_features = self.model.encode_image(images)
            # import pdb;pdb.set_trace()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            score = image_features.unsqueeze(1).repeat([1, self.text_embeddings.size(-2), 1]) * self.text_embeddings
            pred = score.sum(dim=-1).topk(1, dim=1)[1].reshape(1, -1).squeeze(dim=0)
            corr_pred += torch.eq(pred, answer).sum()
            total_num += images.size(0)
        return corr_pred / total_num
    @torch.no_grad()
    def test_epoch_ours(self, dataloader):
        corr_pred = 0
        total_num = 0
        for i, (images, depth, answer) in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
            # import pdb; pdb.set_trace()
            images = images.cuda()
            answer = answer.cuda()
            # import pdb;pdb.set_trace()
            image_features = self.model.visual(images, depth,pos_embed=self.pos_embed)
            # image_features = self.model.encode_image(images)
            # import pdb;pdb.set_trace()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            score = image_features.unsqueeze(1).repeat([1, 4, 1]) * self.text_embeddings
            pred = score.sum(dim=-1).topk(1, dim=1)[1].reshape(1, -1).squeeze(dim=0)
            correct_indices = torch.eq(pred, answer).nonzero(as_tuple=True)[0]        
            # 打印预测正确的下标
            print(f"pedict:{pred}")
            print(f"answer:{answer}")
            print(f"Correctly predicted indices: {correct_indices}")
            corr_pred += torch.eq(pred, answer).sum()
            total_num += images.size(0)
        return corr_pred / total_num

    @torch.no_grad()
    def test_epoch(self, dataloader):
        temp_corr_dict = dict()
        for i, (images, masks, target) in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
            images = images.cuda()
            target = target.cuda()
            image_features = self.model.visual(images, masks,pos_embed=self.pos_embed)
            # image_features = self.model.visual(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            score = torch.matmul(image_features, self.text_embeddings)
            pred = score.topk(1, dim=1)[1].squeeze(dim=1)
            pred_5 = score.topk(5, dim=1)[1].squeeze(dim=1)
            for i in range(target.shape[0]):
                if target[i].item() not in temp_corr_dict:
                    temp_corr_dict[target[i].item()] = [0, 0, 0]
                temp_corr_dict[target[i].item()][0] += 1
                if target[i].item() == pred[i].item():
                    temp_corr_dict[target[i].item()][1] += 1
                if target[i].item() in pred_5[i].tolist():
                    temp_corr_dict[target[i].item()][2] += 1
        return temp_corr_dict
    
    def test(self, epoch=0):
        self.model.visual.eval()
        testset = Imagenet_S()
        self.text_embeddings = zeroshot_classifier(testset.classes, simple_templates, self.model, self.local_rank)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, sampler=sampler, num_workers=16, pin_memory=True)
        with torch.no_grad():
            temp_corr_dict = self.test_epoch(testloader)
            if is_dist_avail_and_initialized():
                output = [None] * dist.get_world_size()
                dist.all_gather_object(output, temp_corr_dict)
            else:
                output = [temp_corr_dict]
            if self.local_rank == 0:
                final_dict = dict()
                for dic in output:
                    for k, v in dic.items():
                        if k not in final_dict.keys():
                            final_dict[k] = v
                        else:
                            final_dict[k][0] += v[0]
                            final_dict[k][1] += v[1]
                            final_dict[k][2] += v[2]
                acc1 = 0.0
                acc5 = 0.0
                num_class = 0
                for v in final_dict.values():
                    acc1 += v[1] / v[0]
                    acc5 += v[2] / v[0]
                    num_class += 1
                acc1 = acc1 / num_class
                acc5 = acc5 / num_class
                print("=====================================")
                print(f"test mean of per class acc-1 step 0: {acc1}")
                print(f"test mean of per class acc-5 step 0: {acc5}")
                print("=====================================")
        return
    @torch.no_grad()
    def test_blink(self):
        totalacc=0
        count = 0
        self.model.visual.eval()
        testset = BLINK_Benchmark('none_spatial.json')
        self.text_embeddings = zeroshot_classifier_ours(testset.captions, self.model, self.local_rank)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, sampler=sampler, num_workers=16, pin_memory=True)
        with torch.no_grad():
            acc1 = self.test_epoch_blink(testloader)
            print("=====================================")
            print(f"BLINK : {acc1}")
            print("=====================================")
        totalacc += acc1*len(testset)
        count += len(testset)
        return
    @torch.no_grad()
    def test_depth_indoor(self):
        totalacc=0
        count = 0
        self.model.visual.eval()
        testset = RGBD_Indoor_Benchmark('none_spatial.json')
        self.text_embeddings = zeroshot_classifier_ours(testset.captions, self.model, self.local_rank)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, sampler=sampler, num_workers=16, pin_memory=True)
        with torch.no_grad():
            acc1 = self.test_epoch_ours(testloader)
            print("=====================================")
            print(f"RGBD class non_spatial : {acc1}")
            print("=====================================")
        totalacc += acc1*len(testset)
        count += len(testset)
        #----------------------------
        testset = RGBD_Indoor_Benchmark('orientation.json')
        self.text_embeddings = zeroshot_classifier_ours(testset.captions, self.model, self.local_rank)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, sampler=sampler, num_workers=16, pin_memory=True)
        with torch.no_grad():
            acc1 = self.test_epoch_ours(testloader)
            print("=====================================")
            print(f"RGBD class orientation : {acc1}")
            print("=====================================")
        totalacc += acc1*len(testset)
        count += len(testset)
        #----------------------------
        testset = RGBD_Indoor_Benchmark('relative_depth.json')
        self.text_embeddings = zeroshot_classifier_ours(testset.captions, self.model, self.local_rank)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, sampler=sampler, num_workers=16, pin_memory=True)
        with torch.no_grad():
            acc1 = self.test_epoch_ours(testloader)
            print("=====================================")
            print(f"RGBD class relative_depth : {acc1}")
            print("=====================================")
        totalacc += acc1*len(testset)
        count += len(testset)
        #----------------------------
        testset = RGBD_Indoor_Benchmark('relative_size.json')
        self.text_embeddings = zeroshot_classifier_ours(testset.captions, self.model, self.local_rank)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, sampler=sampler, num_workers=16, pin_memory=True)
        with torch.no_grad():
            acc1 = self.test_epoch_ours(testloader)
            print("=====================================")
            print(f"RGBD class relative_size : {acc1}")
            print("=====================================")
        totalacc += acc1*len(testset)
        count += len(testset)
        #----------------------------
        testset = RGBD_Indoor_Benchmark('spatial_relation.json')
        self.text_embeddings = zeroshot_classifier_ours(testset.captions, self.model, self.local_rank)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, sampler=sampler, num_workers=16, pin_memory=True)
        with torch.no_grad():
            acc1 = self.test_epoch_ours(testloader)
            print("=====================================")
            print(f"RGBD class spatial_relation : {acc1}")
            print("=====================================")
        totalacc += acc1*len(testset)
        count += len(testset)
        #----------------------------
        print("total acc:",totalacc/count)
        return
    @torch.no_grad()
    def test_depth_indoor_spatial(self):
        totalacc=0
        count = 0
        self.model.visual.eval()
        testset = RGBD_Indoor_Benchmark('spatial_relation.json')
        self.text_embeddings = zeroshot_classifier_ours(testset.captions, self.model, self.local_rank)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, sampler=sampler, num_workers=16, pin_memory=True)
        with torch.no_grad():
            acc1 = self.test_epoch_ours(testloader)
            print("=====================================")
            print(f"RGBD class spatial_relation : {acc1}")
            print("=====================================")
        totalacc += acc1*len(testset)
        count += len(testset)
        #----------------------------
        print("total acc:",totalacc/count)
        return
    @torch.no_grad()
    def test_depth_outdoor(self):
        totalacc=0
        count=0
        self.model.visual.eval()
        testset = RGBD_Outdoor_Benchmark('/home/aiops/wangzh/data/RGBD-benchmark/out_doors','non_spatial.json')
        self.text_embeddings = zeroshot_classifier_ours(testset.captions, self.model, self.local_rank)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, sampler=sampler, num_workers=16, pin_memory=True)
        with torch.no_grad():
            acc1 = self.test_epoch_ours(testloader)
            print("=====================================")
            print(f"RGBD class non_spatial : {acc1}")
            print("=====================================")
        totalacc += acc1*42
        # ------------------
        testset = RGBD_Outdoor_Benchmark('/home/aiops/wangzh/data/RGBD-benchmark/out_doors','object_orientation.json')
        self.text_embeddings = zeroshot_classifier_ours(testset.captions, self.model, self.local_rank)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, sampler=sampler, num_workers=16, pin_memory=True)
        with torch.no_grad():
            acc1 = self.test_epoch_ours(testloader)
            print("=====================================")
            print(f"RGBD class object_orientation : {acc1}")
            print("=====================================")
        totalacc += acc1*36
        # ------------------
        testset = RGBD_Outdoor_Benchmark('/home/aiops/wangzh/data/RGBD-benchmark/out_doors','relative_depth.json')
        self.text_embeddings = zeroshot_classifier_ours(testset.captions, self.model, self.local_rank)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, sampler=sampler, num_workers=16, pin_memory=True)
        with torch.no_grad():
            acc1 = self.test_epoch_ours(testloader)
            print("=====================================")
            print(f"RGBD class relative_depth : {acc1}")
            print("=====================================")
        totalacc += acc1*47
        # ------------------
        testset = RGBD_Outdoor_Benchmark('/home/aiops/wangzh/data/RGBD-benchmark/out_doors','relative_size.json')
        self.text_embeddings = zeroshot_classifier_ours(testset.captions, self.model, self.local_rank)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, sampler=sampler, num_workers=16, pin_memory=True)
        with torch.no_grad():
            acc1 = self.test_epoch_ours(testloader)
            print("=====================================")
            print(f"RGBD class relative_size : {acc1}")
            print("=====================================")
        totalacc += acc1*27
        # ------------------
        testset = RGBD_Outdoor_Benchmark('/home/aiops/wangzh/data/RGBD-benchmark/out_doors','relative_spatial_position.json')
        self.text_embeddings = zeroshot_classifier_ours(testset.captions, self.model, self.local_rank)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, sampler=sampler, num_workers=16, pin_memory=True)
        with torch.no_grad():
            acc1 = self.test_epoch_ours(testloader)
            print("=====================================")
            print(f"RGBD class relative_spatial_position : {acc1}")
            print("=====================================")
        totalacc += acc1*49
        totalacc /= (42+36+47+27+49)
        print(f"all answer:{totalacc}")

        return
    @torch.no_grad()
    def test_depth(self,epoch=0):
        self.model.visual.eval()
        testset = RGBD_Benchmark_Test('/home/aiops/wangzh/data/clip-depth/RGBD-benchmark')
        self.text_embeddings = zeroshot_classifier_ours(testset.captions, self.model, self.local_rank)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, sampler=sampler, num_workers=16, pin_memory=True)
        with torch.no_grad():
            acc1 = self.test_epoch_ours(testloader)
            print("=====================================")
            print(f"test mean of  RGBD class acc-1 : {acc1}")
            print("=====================================")
        return
    @torch.no_grad()
    def test_depth2(self,epoch=0):
        self.model.visual.eval()
        testset = RGBD_Benchmark_Test2('/home/aiops/wangzh/data/clip-depth/RGBD-benchmark')
        self.text_embeddings = zeroshot_classifier_ours(testset.captions, self.model, self.local_rank)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, sampler=sampler, num_workers=16, pin_memory=True)
        with torch.no_grad():
            acc1 = self.test_epoch_ours(testloader)
            print("=====================================")
            print(f"test mean of  RGBD-2 class acc-1 : {acc1}")
            print("=====================================")
        return
    @torch.no_grad()
    def test_flickr(self,epoch=0):
        self.model.visual.eval()
        text_list = []
        feature_list = []
        with torch.no_grad():
            with open("/home/aiops/wangzh/data/flickr30k/results_20130124.token", 'r') as f:
                dataset = f.readlines()
                for data in dataset:
                    image = data.split('\t')[0]
                    text = data.split('\t')[1]
                    text_list.append(text)
            len_list = len(text_list)
            print(len_list)
            for i in range(20):
                text = text_list[i*len_list//20: (i+1)*len_list//20]
                text = clip.tokenize(text, truncate=True).to('cuda')
                feature_list.append(self.model.encode_text(text))
            text_feature = torch.concatenate(feature_list, dim=0)

            data_root = "/home/aiops/wangzh/data/flickr30k/flickr30k-images"
            depth_root = "/home/aiops/wangzh/data/flickr30k/flick30k-depth-new"
            print("begin image")
            img_feature_list = []
            with open("/home/aiops/wangzh/data/flickr30k/results_20130124.token", 'r') as f:
                dataset = f.readlines()
                data_len = len(dataset)
                for i in range(data_len//5):
                    #1 image corresponding to 5 captions
                    data = dataset[5*i]
                    image_name = data.split('\t')[0][:-2]
                    # import pdb;pdb.set_trace()
                    image = Image.open(os.path.join(data_root,image_name)).convert('RGB')
                    image = clip_transform(image).unsqueeze(0).to('cuda')
                    depth = Image.open(os.path.join(depth_root,image_name)).convert('L')
                    depth = depth_transform(depth).unsqueeze(0).to('cuda')
                    # import pdb;pdb.set_trace()
                    img_feature = self.model.visual(image,depth,pos_embed=self.pos_embed)
                    img_feature_list.append(img_feature)
                    torch.cuda.empty_cache()
                    del img_feature, image
            image_feature = torch.concatenate(img_feature_list, dim=0)
        print("begin process")
        get_accuracy_i2t(text_feature, image_feature, 1)
        get_accuracy_i2t(text_feature, image_feature, 5)
        # get_accuracy_i2t(text_feature, image_feature, 10)
        get_accuracy_t2i(text_feature, image_feature, 1)
        get_accuracy_t2i(text_feature, image_feature, 5)
        # get_accuracy_t2i(text_feature, image_feature, 10)
        return None


    @torch.no_grad()
    def test_epoch_scannet(self, dataloader):

        self.model.visual.eval()
        # testset = RGBD_Benchmark_Test('/home/maiyubo/llm/AlphaCLIP-main/train/dataset/data/rgbd_benchmark')
        testset = ScanRefer_Test('/home/aiops/wangzh/data/scanner/scannet_2d_HR3', self.model)
        # self.text_embeddings = zeroshot_classifier_ours(testset.captions, self.model, self.local_rank)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        dataloader = torch.utils.data.DataLoader(testset, batch_size=2, sampler=sampler, num_workers=0)
        
        corr_pred = 0
        total_num = 0
        for i, (images, depth, text_embeddings, answer) in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
            images = images.cuda()
            answer = answer.cuda()
            # image_features = self.model.encode_image(images)
            # import pdb;pdb.set_trace()
            image_features = self.model.visual(images, depth,pos_embed=self.pos_embed)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            score = image_features.unsqueeze(1).repeat([1, 4, 1]) * text_embeddings
            pred = score.sum(dim=-1).topk(1, dim=1)[1].reshape(1, -1).squeeze(dim=0)
            corr_pred += torch.eq(pred, answer).sum()
            total_num += images.size(0)
        return corr_pred / total_num

    @torch.no_grad()
    def test_epoch_scannet2(self, dataloader):
        
        self.model.visual.eval()
        # testset = RGBD_Benchmark_Test('/home/maiyubo/llm/AlphaCLIP-main/train/dataset/data/rgbd_benchmark')
        testset = ScanRefer_Test2('/home/aiops/wangzh/data/scanner/scannet_2d_HR3', self.model)
        # self.text_embeddings = zeroshot_classifier_ours(testset.captions, self.model, self.local_rank)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        dataloader = torch.utils.data.DataLoader(testset, batch_size=300, sampler=sampler, num_workers=0)
        
        corr_pred = 0
        total_num = 0
        for i, (images, depth, text_embeddings, answer) in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
            images = images.cuda()
            answer = answer.cuda()
            # image_features = self.model.encode_image(images)
            image_features = self.model.visual(images, depth,pos_embed=self.pos_embed)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            score = image_features.unsqueeze(1).repeat([1, 4, 1]) * text_embeddings
            pred = score.sum(dim=-1).topk(1, dim=1)[1].reshape(1, -1).squeeze(dim=0)
            corr_pred += torch.eq(pred, answer).sum()
            total_num += images.size(0)
        return corr_pred / total_num
    
    @torch.no_grad()
    def test_epoch_scannet3d(self, dataloader):
        
        self.model.visual.eval()
        # testset = RGBD_Benchmark_Test('/home/maiyubo/llm/AlphaCLIP-main/train/dataset/data/rgbd_benchmark')
        testset = ScanRefer_Testnr3d('/home/aiops/wangzh/data/scanner/scannet_2d_HR3', self.model)
        # self.text_embeddings = zeroshot_classifier_ours(testset.captions, self.model, self.local_rank)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        dataloader = torch.utils.data.DataLoader(testset, batch_size=300, sampler=sampler, num_workers=0)
        
        corr_pred = 0
        total_num = 0
        for i, (images, depth, text_embeddings, answer) in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
            images = images.cuda()
            answer = answer.cuda()
            # image_features = self.model.encode_image(images)
            image_features = self.model.visual(images, depth,pos_embed=self.pos_embed)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            score = image_features.unsqueeze(1).repeat([1, 4, 1]) * text_embeddings
            pred = score.sum(dim=-1).topk(1, dim=1)[1].reshape(1, -1).squeeze(dim=0)
            corr_pred += torch.eq(pred, answer).sum()
            total_num += images.size(0)
        return corr_pred / total_num


    def train(self, common_pair=False, resume=False, amp=False, warmup_length=200):

        test_loaders=dict()
        testset_ours = RGBD_Benchmark_Test('/home/aiops/wangzh/data/clip-depth/RGBD-benchmark')
        coco_dataset = CocoDataset(root="/home/aiops/wangzh/data/val2017", annFile="/home/aiops/wangzh/data/captions_val2017.json", transform=clip_transform,depth_transform=depth_transform)

        # test_sampler = DistributedSampler(dataset=testset, shuffle=True)
        # test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, sampler=test_sampler, num_workers=16, pin_memory=True)
        for name, testset in zip(['COCO', 'OURS'], [coco_dataset, testset_ours]):
            # test_sampler = DistributedSampler(dataset=testset, shuffle=True)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=256, num_workers=4, pin_memory=True)
            test_loaders[name] = test_loader   


        # trainset=COCO_Masked_Test(hi_res=True)
        trainset = DepthDataset( transform=clip_transform, depth_transform=depth_transform)
        # import pdb;pdb.set_trace()
        train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)
        self.scheduler = cosine_lr(self.optimizer, base_lr=self.lr, warmup_length=warmup_length, steps=10000, para_gamma=self.para_gamma)
        start_epoch = 0
        resume_iter = 0
        if resume == True and os.listdir(self.ckptdir) != []:
            resume_pth = os.listdir(self.ckptdir).sort(key=lambda x: int(x[5:-4]))[-1]
            resume_iter = int(resume_pth[5:-4])
            start_epoch = resume_iter // len(train_loader)
            map_location = {'cuda:0': f'cuda:{self.local_rank}'}
            self.model.visual.load_state_dict(torch.load(os.path.join(self.ckptdir, resume_pth), map_location=map_location),strict=False)
            print(f"load resumed checkpoint: {resume_pth}")  
        for epoch in range(start_epoch, self.num_epoch):
            # train
            loss = self.train_epoch(train_loader, test_loaders, epoch, start_iter=resume_iter, amp=amp)

        # Finish wandb run



def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29992"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return rank % num_gpus

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--lr', default=4e-5, type=float, help='lr.')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='wd.')
    parser.add_argument('--log_scale', default=4.6052, type=float, help='clip temperature log scale.')
    parser.add_argument('--lora_rank', default=-1, type=int, help='lora rank (-1 to not use lora).')
    parser.add_argument('--common_pair', default=0.0, type=float, help='propotion to use image with all 1 alpha and whole caption.')
    parser.add_argument('--para_gamma', default=0.01, type=float, help='para_gamma of other parameters')
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    parser.add_argument("--amp", action="store_true", help="bf16 taining.")
    parser.add_argument("--exp_name", default="auto", type=str, help="specify experiment name.")
    parser.add_argument("--warmup_length", default=200, type=int, help="warmup_length.")
    parser.add_argument("--epoch_num", default=40, type=int, help="number of epochs.")
    parser.add_argument("--subnum", default=1e4, type=float, help="sub data number.")
    parser.add_argument("--pos_embed", default=None, type=str, help="position method.")
    
    args = parser.parse_args()
    local_rank = setup_distributed()
    trainer = CLIP_Clean_Train(
        local_rank=local_rank, 
        lr=args.lr, 
        weigth_decay=args.weight_decay, 
        log_scale=args.log_scale, 
        lora_rank=args.lora_rank, 
        common_pair=args.common_pair, 
        para_gamma=args.para_gamma,
        exp_name=args.exp_name,
        warmup_length=args.warmup_length,
        epoch_num=args.epoch_num,
        subnum=int(args.subnum),
        pos_embed=args.pos_embed
    )
    # trainer.train(common_pair=args.common_pair, resume=args.resume, amp=args.amp, warmup_length=args.warmup_length)
    # train=DepthDataset()
    # trainer.test_blink()
    # depth_acc=trainer.test_depth()
    # depth_acc=trainer.test_depth2()
    # print("outdoor:")
    # trainer.test_depth_outdoor()
    # print("indoor:")
    # trainer.test_depth_indoor_spatial()
    # scan_acc=trainer.test_epoch_scannet(None)
    # print("scan:",scan_acc)
    # scan2_acc=trainer.test_epoch_scannet2(None)
    # print("scan2:",scan2_acc)
    acc=trainer.test_epoch_retrieval(None)
    # print("coco:",acc)
    acc=trainer.test_flickr()
    # acc=trainer.test_imagenet()

    # scan3d_acc=trainer.test_epoch_scannet3d(None)
    # print("scan3d",scan3d_acc)
