import torch
import alpha_clip
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
from dataset.mask_image_test import ScanRefer_Test
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

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
from dataset.mask_image_test import RGBD_Benchmark_Test
from torchvision.datasets import CocoCaptions
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
simple_templates = [
    'a photo of a {}.'
]

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
        self.print_trainable_params()
        checkpoint=os.listdir(self.ckptdir)
        checkpoint.sort(key=lambda x: int(x[5:-4]))
        # resume_pth = os.listdir(self.ckptdir).sort(key=lambda x: int(x[5:-4]))[-1]
        # resume_pth=checkpoint[-1]

        resume_pth ="iter_30000.pth"
        map_location = {'cuda:0': f'cuda:{self.local_rank}'}
        self.model.visual.load_state_dict(torch.load(os.path.join(self.ckptdir, resume_pth), map_location=map_location),strict=False)
        print(f"load resumed checkpoint:{self.ckptdir} {resume_pth}")  





    def print_trainable_params(self):
        print("Trainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
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

    @torch.no_grad()
    def test_epoch_retrieval(self, dataloader):
        image_features = []
        text_features = []
        coco_dataset = CocoDataset(root="/home/aiops/wangzh/data/val2017", annFile="/home/aiops/wangzh/data/captions_val2017.json", transform=clip_transform,depth_transform=depth_transform)
        dataloader = torch.utils.data.DataLoader(coco_dataset, batch_size=256, num_workers=4, pin_memory=True)

        checkpoint=os.listdir(self.ckptdir)
        checkpoint.sort(key=lambda x: int(x[5:-4]))
        # resume_pth = os.listdir(self.ckptdir).sort(key=lambda x: int(x[5:-4]))[-1]
        # resume_pth=checkpoint[-1]
        resume_pth ="iter_30000.pth"
        map_location = {'cuda:0': f'cuda:{self.local_rank}'}
        self.model.visual.load_state_dict(torch.load(os.path.join(self.ckptdir, resume_pth), map_location=map_location),strict=False)
        print(f"load resumed checkpoint:{self.ckptdir} {resume_pth}")  

        with torch.no_grad():
            for images, depth,captions_batch in dataloader:
                images = images.to('cuda')
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
    def test_epoch_ours(self, dataloader):
        corr_pred = 0
        total_num = 0
        for i, (images, depth, answer) in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
            images = images.cuda()
            answer = answer.cuda()
            image_features = self.model.visual(images, depth,pos_embed=self.pos_embed)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            score = image_features.unsqueeze(1).repeat([1, 4, 1]) * self.text_embeddings
            pred = score.sum(dim=-1).topk(1, dim=1)[1].reshape(1, -1).squeeze(dim=0)
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
    def test_depth(self,epoch=0):
        checkpoint=os.listdir(self.ckptdir)
        checkpoint.sort(key=lambda x: int(x[5:-4]))
        # resume_pth = os.listdir(self.ckptdir).sort(key=lambda x: int(x[5:-4]))[-1]
        # resume_pth=checkpoint[-1]
        resume_pth ="iter_30000.pth"
        map_location = {'cuda:0': f'cuda:{self.local_rank}'}
        self.model.visual.load_state_dict(torch.load(os.path.join(self.ckptdir, resume_pth), map_location=map_location),strict=False)
        print(f"load resumed checkpoint:{self.ckptdir} {resume_pth}")  
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
            depth_root = "/home/aiops/wangzh/data/flickr30k/flickr-depth"
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
                    img_feature = self.model.visual(image,depth,pos_embed=self.pos_embed)
                    img_feature_list.append(img_feature)
                    torch.cuda.empty_cache()
                    del img_feature, image
            image_feature = torch.concatenate(img_feature_list, dim=0)
        print("begin process")
        get_accuracy_i2t(text_feature, image_feature, 1)
        get_accuracy_i2t(text_feature, image_feature, 5)
        get_accuracy_i2t(text_feature, image_feature, 10)
        get_accuracy_t2i(text_feature, image_feature, 1)
        get_accuracy_t2i(text_feature, image_feature, 5)
        get_accuracy_t2i(text_feature, image_feature, 10)
        return None


    @torch.no_grad()
    def test_epoch_scannet(self, dataloader):
        checkpoint=os.listdir(self.ckptdir)
        checkpoint.sort(key=lambda x: int(x[5:-4]))
        # resume_pth = os.listdir(self.ckptdir).sort(key=lambda x: int(x[5:-4]))[-1]
        # resume_pth=checkpoint[-1]
        resume_pth ="iter_30000.pth"
        map_location = {'cuda:0': f'cuda:{self.local_rank}'}
        # self.model.visual.load_state_dict(torch.load(os.path.join(self.ckptdir, resume_pth), map_location=map_location),strict=False)
        # print(f"load resumed checkpoint:{self.ckptdir} {resume_pth}")  
        
        # models, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
        # tokenizer = get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
        # self.model=models.to('cuda')

        self.model.visual.eval()
        # testset = RGBD_Benchmark_Test('/home/maiyubo/llm/AlphaCLIP-main/train/dataset/data/rgbd_benchmark')
        testset = ScanRefer_Test('/home/aiops/wangzh/data/scanner/scannet_2d_HR3', self.model)
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
    def test_epoch_scannet2(self, dataloader):
        checkpoint=os.listdir(self.ckptdir)
        checkpoint.sort(key=lambda x: int(x[5:-4]))
        # resume_pth = os.listdir(self.ckptdir).sort(key=lambda x: int(x[5:-4]))[-1]
        # resume_pth=checkpoint[-1]
        resume_pth ="iter_30000.pth"
        map_location = {'cuda:0': f'cuda:{self.local_rank}'}
        self.model.visual.load_state_dict(torch.load(os.path.join(self.ckptdir, resume_pth), map_location=map_location),strict=False)
        print(f"load resumed checkpoint:{self.ckptdir} {resume_pth}")  
        
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


    def train(self, common_pair=False, resume=False, amp=False, warmup_length=200):
        # testset_image_s = Imagenet_S(hi_res=True)
        # testset_image_s_all_one = Imagenet_S(hi_res=True, all_one=True)
        # testset_coco = COCO_Masked_Test(hi_res=True)
        # trainset = Alpha_GRIT(common_pair=common_pair)
        # trainset = Alpha_GRIT(ids_file='grit_1m_ids.pkl', root_pth='grit-1m/', common_pair=common_pair, subnum=self.subnum, hi_res=True)
        # trainset = torch.utils.data.ConcatDataset(datasets=[trainset, ImageNet_Masked()])
        # test_loaders = dict()
        # for name, testset in zip(['COCO', 'Imagenet-S', 'Imagenet-S_all_one'], [testset_coco, testset_image_s, testset_image_s_all_one]):
        #     test_sampler = DistributedSampler(dataset=testset, shuffle=True)
        #     test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, sampler=test_sampler, num_workers=16, pin_memory=True)
        #     test_loaders[name] = test_loader   
        # os.environ['WANDB_MODE'] = 'offline'
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
    # scan_acc=trainer.test_epoch_scannet(None)
    # scan2_acc=trainer.test_epoch_scannet2(None)
    # acc=trainer.test_epoch_retrieval(None)
    # depth_acc=trainer.test_depth()
    acc=trainer.test_flickr()
    # print("scan:",scan_acc)
    # print("scan2:",scan2_acc)
    # print("coco:",acc)
    # print("rgbd:",depth_acc)