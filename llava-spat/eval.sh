source /home/aiops/wangzh/miniconda3/bin/activate
conda activate llava
CUDA_VISIBLE_DEVICES=0 python -m rgbd_eval.py