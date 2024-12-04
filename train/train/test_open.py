import torch
from PIL import Image
import open_clip
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将 'train' 目录添加到 sys.path 中
sys.path.append(parent_dir)
from dataset.mask_image_test import RGBD_Benchmark_Test,RGBD_Benchmark_Test2,RGBD_Outdoor_Benchmark
from torch.utils.data import Dataset
from collections import defaultdict
import json
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler
from tqdm import tqdm
import argparse
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

testset = RGBD_Outdoor_Benchmark('/home/aiops/wangzh/data/RGBD-benchmark/out_doors','non_spatial.json')

sampler = SequentialSampler(testset)
dataloader = torch.utils.data.DataLoader(testset, batch_size=1, sampler=sampler, num_workers=0)

# model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
# model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
# tokenizer = open_clip.get_tokenizer('ViT-L-14-336')

# image = preprocess(Image.open("/home/aiops/wangzh/data/sa_1953851.jpg")).unsqueeze(0)
# text = tokenizer(["a diagram", "a dog", "a cat"])

# with torch.no_grad(), torch.cuda.amp.autocast():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)

#     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]


def test_epoch_scannet(tasks):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')

    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    # tokenizer = open_clip.get_tokenizer('ViT-L-14-336')

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

        image = preprocess(image).unsqueeze(0)
        text = tokenizer(caption)
        # import pdb;pdb.set_trace()
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = torch.sigmoid(text_probs)
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