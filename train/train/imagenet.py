imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

import os

import torch
from torchvision import datasets


class ImageNetCategory():
    """
        For ImageNet-like directory structures without sessions/conditions:
        .../{category}/{img_name}
    """

    def __init__(self):
        pass

    def __call__(self, full_path):
        img_name = full_path.split("/")[-1]
        category = full_path.split("/")[-2]
        return category


class ImageNetDataset(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __init__(self, *args, **kwargs):
        super(ImageNetDataset, self).__init__(*args, **kwargs)

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        sample, target = super(ImageNetDataset, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        new_target = ImageNetCategory()(path)
        original_tuple = (sample, new_target)
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class ImageNetClipDataset(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder

    Adapted from:
    https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """
    SOFT_LABELS = "soft_labels"
    HARD_LABELS = "hard_labels"

    def __init__(self, label_type, mappings, *args, **kwargs):
        self.label_type = label_type
        self.clip_class_mapping = mappings
        super(ImageNetClipDataset, self).__init__(*args, **kwargs)

    def _get_new_template_hard_labels(self, image_path):
        file_name = os.path.basename(image_path)
        target_class = self.clip_class_mapping[file_name]
        target_index = self.class_to_idx[target_class]
        return target_index

    def _get_new_template_soft_labels(self, image_path):
        file_name = os.path.basename(image_path)
        target_class = self.clip_class_mapping[file_name]
        return target_class

    def __getitem__(self, index):
        """override the __getitem__ method. This is the method that dataloader calls."""
        # this is what ImageFolder normally returns
        (sample, target) = super(ImageNetClipDataset, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]
        if self.label_type == ImageNetClipDataset.HARD_LABELS:
            new_target = self._get_new_template_hard_labels(path)
        elif self.label_type == ImageNetClipDataset.SOFT_LABELS:
            new_target = self._get_new_template_soft_labels(path)
        else:
            new_target = target
        new_target = get_label(new_target)
        original_tuple = (sample, new_target,)
        return original_tuple

def get_label(fold_name):
    with open("/home/aiops/wangzh/zss/AlphaCLIP/train/train/categories.txt", "r", encoding='utf-8') as f:
        data = f.readlines()
        #print(len(data))
        for i in range(len(data)):
            if data[i][:9] == fold_name:
                return torch.tensor([i])





def imagenet_data_loader(transform):
    imagenet_data = ImageNetDataset('/home/aiops/wangzh/data/imagenet-val', transform)
    data_loader = torch.utils.data.DataLoader(
        imagenet_data,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )
    return data_loader, imagenet_data

if __name__ == "__main__":
    print(get_label("n03584254"))