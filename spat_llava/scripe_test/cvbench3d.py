import argparse
import torch
import os
import json
from tqdm import tqdm

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    # import pdb;pdb.set_trace()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    model=model.to(torch.float16)
    # import pdb;pdb.set_trace()

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode
    
    base_path = args.base_path
    answers = []
    # with open(os.path.join(base_path, "annotations.json"), "r") as reader:
    # with open('/home/aiops/wangzh/data/blink/BLINK/Counting/output.json', "r") as reader:
    # with open('/home/aiops/wangzh/data/blink/BLINK/Visual_Correspondence/output.json', "r") as reader:
    # with open('/home/aiops/wangzh/data/blink/BLINK/Jigsaw/output.json', "r") as reader:
    # with open('/home/aiops/wangzh/data/CV-Bench/test3d-depth.jsonl', "r") as reader:
    with open('/home/aiops/wangzh/data/CV-Bench/test3d-depth.jsonl', "r") as reader:
    # with open('/home/aiops/wangzh/data/blink/BLINK/Multi-view_Reasoning/output.json', "r") as reader:
    # with open('/home/aiops/wangzh/data/blink/BLINK/Object_Localization/output.json', "r") as reader:
    # with open('/home/aiops/wangzh/data/blink/BLINK/Relative_Depth/output.json', "r") as reader:
    # with open('/home/aiops/wangzh/data/blink/BLINK/Spatial_Relation/output.json', "r") as reader:
    # with open('/home/aiops/wangzh/data/RGBD-benchmark/out_doors/all.json', "r") as reader:
    # with open('/home/aiops/wangzh/data/realworldqa/updated.json', "r") as reader:
    # with open('/home/aiops/wangzh/data/RGBD-benchmark/out_doors/object_orientation.json', "r") as reader:
    # with open('/home/aiops/wangzh/data/RGBD-benchmark/out_doors/relative_depth.json', "r") as reader:
    # with open('/home/aiops/wangzh/data/RGBD-benchmark/out_doors/relative_size.json', "r") as reader:
    # with open('/home/aiops/wangzh/data/RGBD-benchmark/out_doors/relative_spatial_position.json', "r") as reader:
    # with open('/home/aiops/wangzh/data/scanner/indoor-new/all.json', "r") as reader:
    # with open('/home/aiops/wangzh/data/scanner/indoor/orientation.json', "r") as reader:
    # with open('/home/aiops/wangzh/data/scanner/indoor/relative_depth.json', "r") as reader:
    # with open('/home/aiops/wangzh/data/scanner/indoor/relative_size.json', "r") as reader:
    # with open('/home/aiops/wangzh/data/scanner/indoor/spatial_relation.json', "r") as reader:
        # data = json.load(reader)
        for line in tqdm(reader):

        # for line in tqdm(data):
            data = json.loads(line.strip())
            # path1 = os.path.join('/home/aiops/wangzh/data/blink/BLINK/Jigsaw/output_images',data['image_1'])
            # path2 = os.path.join('/home/aiops/wangzh/data/blink/BLINK/Jigsaw/output_images',data['image_2'])
            # path3 = os.path.join('/home/aiops/wangzh/data/blink/BLINK/Jigsaw/output_images',data['image_3'])
            # args.image_file = f"{path1},{path2},{path3}"
            # path1 = os.path.join('/home/aiops/wangzh/data/blink/BLINK/Visual_Correspondence/output_images',data['image_1'])
            # path2 = os.path.join('/home/aiops/wangzh/data/blink/BLINK/Visual_Correspondence/output_images',data['image_2'])
            # args.image_file = f"{path1},{path2}"
            # args.image_file = os.path.join('/home/aiops/wangzh/data/blink/BLINK/Spatial_Relation/output_images',data['image_1'])
            args.image_file = os.path.join('/home/aiops/wangzh/data/CV-Bench',data['filename'])
            
            qs = f"Please answer the following questions with only one  :{data['prompt']}"
            # qs = f"{data['prompt']}Make a choice and explain it."
            # qs = data['prompt']
            # qs = f'''You are given a question with four possible answers labeled as (A), (B), (C),and (D). Please read the question carefully and choose the most appropriate answer by responding with the corresponding letter (A, B, C, or D) only. Do not provide any additional explanation or text.
            # {data['prompt']}'''

            # import pdb;pdb.set_trace()
            # scen = line['scene_id']
            # args.image_file = os.path.join('/home/aiops/wangzh/data/realworldqa/output_images',data['image'])
            # args.image_file = os.path.join('/home/aiops/wangzh/data/RGBD-benchmark/out_doors/pic_all',line['image'])
            # args.image_file = os.path.join('/home/aiops/wangzh/data/scanner/scannet_2d_HR3',scen,'color',line['image'])
            # qs = f'''You will see an image along with four corresponding descriptions (captions). Please carefully observe the image and select the description that best matches the content of the image. Choose one option from (A), (B), (C), or (D).
            #         Options: (A){line['captions'][0]}\n(B){line['captions'][1]}\n(C){line['captions'][2]}\n(D){line['captions'][3]}\nPlease provide your answer with only one of the options and nothing else.'''
            # qs = data['question']

            # args.image_file = os.path.join(base_path, data["image"])
            # qs = data["question"]
            
            # qs = args.query
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in qs:
                if model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            image_files = image_parser(args)
            images = load_images(image_files)
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)

            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            # import pdb;pdb.set_trace()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip().replace("\n", "")
            answers.append(outputs)
            # print(ßoutputs)
        with open(os.path.join('.',f"{model_name}_answers.txt"), "w") as writer:
            writer.writelines([answer + "\n" for answer in answers])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
