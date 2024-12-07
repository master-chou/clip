import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将 'train' 目录添加到 sys.path 中
sys.path.append(parent_dir)
from dataset.mask_image_test import ScanRefer_Test,ScanRefer_Test2,ScanRefer_Testnr3d
from PIL import Image
import requests
import alpha_clip_noall as alpha_clip
from dataset.mask_image_test import RGBD_Benchmark_Test,RGBD_Benchmark_Test2,RGBD_Outdoor_Benchmark
import argparse
from transformers import AutoProcessor, AutoModel
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import json
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler
from tqdm import tqdm
from transformers import CLIPProcessor,CLIPModel
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

clip_standard_transform = Compose([
    ToTensor(), 
    Resize((224, 224), interpolation=Image.BICUBIC),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
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
        self.dataset = RGBD_INDOOR(os.path.join(self.root_dir,'indoor',tasks))
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
        # self.transform = transforms.Compose([
        #     transforms.Resize(224, interpolation=BICUBIC),
        #     transforms.CenterCrop(224),
        #     _convert_image_to_rgb,
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # ])
        # self.depth_transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        # ])

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
        depth = None
        # if self.transform:
        #     image = self.transform(image)
        # if self.depth_transform:
        #     depth = self.depth_transform(depth)
        
        caption = self.dataset.anns[img_ids]
        # texts = alpha_clip.tokenize(caption).cuda()
        # text_embeddings = self.model.encode_text(texts)
        # text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        
        answer = self.dataset.answers[img_ids]
        return image_path, caption, answer


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

# model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
# processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# texts = ["a photo of 2 cats", "a photo of 2 dogs"]
# inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)

# # import pdb;pdb.set_trace()
# logits_per_image = outputs.logits_per_image
# probs = torch.sigmoid(logits_per_image) # these are the probabilities
# print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")

def test_epoch_scannet(tasks):
    # model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to('cuda')
    # processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to('cuda')
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model, _ = alpha_clip.load("ViT-L/14@336px", device='cpu', lora_adapt=False, rank=-1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    import pdb;pdb.set_trace()
    
    model.eval()
    # testset = ScanRefer_Test('/home/aiops/wangzh/data/scanner/scannet_2d_HR3',model)
    testset = RGBD_Outdoor_Benchmark('/home/aiops/wangzh/data/RGBD-benchmark/out_doors','non_spatial.json')
    sampler = SequentialSampler(testset)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1, sampler=sampler, num_workers=0)

    # dataset = ScanRefer(os.path.join('/home/aiops/wangzh/data/scanner/scannet_2d_HR3', 'scanrefer_annotations_all.json'))
    # dataset = RGBD(f'/home/aiops/wangzh/data/RGBD-benchmark/out_doors/{tasks}.json')

    dataset = RGBD_INDOOR(f'/home/aiops/wangzh/data/scanner/indoor/{tasks}.json')

    # dataset = RGBD('/home/aiops/wangzh/data/RGBD-benchmark/out_doors/object_orientation.json')
    # dataset = RGBD('/home/aiops/wangzh/data/RGBD-benchmark/out_doors/relative_depth.json')
    # dataset = RGBD('/home/aiops/wangzh/data/RGBD-benchmark/out_doors/relative_size.json')
    # dataset = RGBD('/home/aiops/wangzh/data/RGBD-benchmark/out_doors/relative_spatial_position.json')
    corr_pred = 0
    total_num = 0
    image_ids = list(dataset.imgs.keys())
    for idx in tqdm(range(len(image_ids))):
    # for i, (images, depth, answer) in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):

    # for idx in tqdm(range(200)):
        # images = images.cuda()
        # answer = answer.cuda()
        img_ids = image_ids[idx]
        # image_path = os.path.join('/home/aiops/wangzh/data/RGBD-benchmark/out_doors/pic_all', dataset.imgs[img_ids])
        image_path = os.path.join('/home/aiops/wangzh/data/scanner/scannet_2d_HR3', dataset.scene_id[img_ids],'color',dataset.imgs[img_ids])

        caption = dataset.anns[img_ids]
        answer = dataset.answers[img_ids]
        image = Image.open(image_path).convert("RGB")

        # import pdb;pdb.set_trace()
        inputs = processor(text=caption, images=image, padding="max_length",truncation=True ,return_tensors="pt").to("cuda")
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = torch.sigmoid(logits_per_image)
        pred = torch.argmax(probs).item()
        # import pdb;pdb.set_trace()
        if pred == answer:
            corr_pred += 1
        total_num += 1
        # image_features = self.model.visual(images, depth,pos_embed=self.pos_embed)
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # score = image_features.unsqueeze(1).repeat([1, 4, 1]) * text_embeddings
        # pred = score.sum(dim=-1).topk(1, dim=1)[1].reshape(1, -1).squeeze(dim=0)
        # corr_pred += torch.eq(pred, answer).sum()
        # total_num += images.size(0)
        # print("correct",corr_pred)
    print("total",total_num)
    return corr_pred / total_num

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',type=str)
    args = parser.parse_args()

    acc=test_epoch_scannet(args.task)
    print(args.task)
    print(acc)