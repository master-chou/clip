from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava_backup import eval_model
# from llava.eval.run_llava import eval_model

model_path = "/home/aiops/wangzh/llava-spat/checkpoints/llava-v1.5-7b-final-neg-lora"


# prompt = "Describe the orientation and position relationship between two giraffes in the picture."
# image_file = "/home/aiops/wangzh/data/RGBD-benchmark/out_doors/pic_all/000000023744.jpg"

# prompt = "Describe the orientation position relationship between two bus in the picture."
# image_file = "/home/aiops/wangzh/data/RGBD-benchmark/out_doors/pic_all/000000017286.jpg"
# prompt = "Describe the spatial relationship between the three stop signs in the picture."
# image_file = "/home/aiops/wangzh/data/RGBD-benchmark/out_doors/pic_all/000000010700.jpg"

# prompt = "Describe the position of the objects in the picture."
# image_file = "/home/aiops/wangzh/data/RGBD-benchmark/out_doors/pic_all/000000016983.jpg"


base_path = "/home/aiops/wangzh/data/RGBD-benchmark"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": '/home/aiops/wangzh/llava/vicuna-7b-v1.5',
    # "model_base": 'lmsys/vicuna-13b-v1.5',
    "model_name": get_model_name_from_path(model_path),
    # "query": prompt,
    "conv_mode": None,
    # "image_file": image_file,
    "base_path": base_path,
    "sep": ",",
    "temperature": 0.5,
    "top_p": 0.7,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)