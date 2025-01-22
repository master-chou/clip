# Spatial-CLIP


## Install

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/master-chou/spatial-clip.git
cd spatial-clip
```

2. Install Package
```Shell
conda create -n spclip python=3.10 -y
conda activate spclip
pip install --upgrade pip  # enable PEP 660 support
pip install -r requirements.txt
```

## Inference

To perform inference using Spatial-CLIP, follow these steps:

1. **Prepare Your Inputs**:
   - Ensure you have the image and text file ready for inference. The image should be the visual input you want to analyze, and the text file should contain the textual input or instructions for the model.

2. **Modify `test.sh`**:
   - Open the `test.sh` file in a text editor.
   - Update the following parameters to match your input paths:
     - Replace `YOUR_IMAGE_PATH` with the path to your image file.
     - Replace `YOUR_TEXT_FILE_PATH` with the path to your text file.

3. **Run the Inference**:
   - Execute the `test.sh` script to run the inference:
     ```bash
     bash test.sh
     ```
   - The script will load the model, process the inputs, and generate the output based on the spatial understanding capabilities of Spatial-CLIP.

This project builds upon and extends the work from [LLaVA](https://github.com/haotian-liu/LLaVA) and [AlphaCLIP](https://github.com/microsoft/AlphaCLIP), incorporating spatial understanding capabilities into vision-language models.

## Acknowledgments

This repository is built upon the following excellent open-source projects:

### LLaVA
- **Repository**: [https://github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
- **Paper**: [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
- **License**: Apache License 2.0

### AlphaCLIP
- **Repository**: [https://github.com/microsoft/AlphaCLIP](https://github.com/microsoft/AlphaCLIP)
- **Paper**: [AlphaCLIP: A CLIP Model Focusing on Wherever You Want](https://arxiv.org/abs/2312.03818)
- **License**: MIT License

## Citations

If you find this project useful, please consider citing our work along with the original papers:

Coming soon.
