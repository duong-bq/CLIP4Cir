import json
import random
from pathlib import Path
from typing import List

import PIL
import PIL.Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

base_path = Path(__file__).absolute().parents[1].absolute()


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class SquarePad:
    """
    Square pad the input image with zero padding
    """

    def __init__(self, size: int):
        """
        For having a consistent preprocess pipeline with CLIP we need to have the preprocessing output dimension as
        a parameter
        :param size: preprocessing output dimension
        """
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


def squarepad_transform(dim: int):
    """
    CLIP-like preprocessing transform on a square padded image
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def targetpad_transform(target_ratio: float, dim: int):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def generate_randomized_fiq_caption(captions, type=-1, split: str = 'train'):
    # Khi validate thì ghép tất cả các option của modified_text
    if split == 'val' or split == 'test':
        if len(captions) == 2:
            caption = f"{captions[0].strip('.?, ').capitalize()} and {captions[1].strip('.?, ')}"
            return caption
        elif len(captions) > 2:
            # Nếu là data của gemini chứa 4 hoặc 5 option thì chỉ random 1 option
            return random.choice(captions)
        else:
            return caption[0]
    else:
        if len(captions) == 2:
            if captions[0] == captions[1]:
                return captions[0]
            
            random_num = random.random()
            if type == 0:
                random_num = 0.12
            elif type == 1:
                random_num = 0.37
            elif type == 2:
                random_num = 0.62
            elif type == 3:
                random_num = 0.88
            if random_num < 0.25:
                caption = f"{captions[0].strip('.?, ')} and {captions[1].strip('.?, ')}"
            elif 0.25 < random_num < 0.5:
                caption = f"{captions[1].strip('.?, ')} and {captions[0].strip('.?, ')}"
            elif 0.5 < random_num < 0.75:
                caption = f"{captions[0].strip('.?, ')}"
            else:
                caption = f"{captions[1].strip('.?, ')}"
            return caption
        elif len(captions) > 2:
            caption = random.choice(captions)
            return caption
        else:
            return captions[0]


class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split: str, dress_types: List[str], mode: str, preprocess: callable, plus: bool = False):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        :param plus: use scaling positive and negative dataset or not
        """
        self.mode = mode
        self.dress_types = dress_types
        self.split = split

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee', 'gemini']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee', 'gemini]")

        self.preprocess = preprocess

        # GET TRIPLETS made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            if split == "train":
                with open(base_path / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.{split}.gemini.json') as f:
                    triplets = json.load(f)
                    triplets = triplets[0:len(triplets) // 2]
                    self.triplets.extend(triplets)
                with open(base_path / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                    triplets = json.load(f)
                    triplets = triplets[len(triplets) // 2:]
                    self.triplets.extend(triplets)
            else:
                with open(base_path / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                    self.triplets.extend(json.load(f))
        if plus and split == 'train':
            # with open(base_path / 'fashionIQ_dataset' / 'captions' / 'cap.extend_clip.train.json') as f:
            #     extended_triplets = json.load(f)
            #     for i, item in enumerate(extended_triplets):
            #         item['captions'] = [item['captions'][0], item['captions'][0]]
            #         extended_triplets[i] = item
            #     self.triplets.extend(extended_triplets)
            with open(base_path / 'fashionIQ_dataset/captions/hoang_processed.json') as f:
                extended_triplets = json.load(f)
                extended_triplets = random.sample(extended_triplets, int(len(extended_triplets) * 0.5))
                self.triplets.extend(extended_triplets)
            print("Use scaling dataset!")

        # GET THE IMAGE NAMES
        self.image_names: list = []
        for dress_type in dress_types:
            with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))
        # if plus and split == "train":
        #     with open(base_path / 'fashionIQ_dataset/image_splits/FLUX_plus.json') as f:
        #         self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                image_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']

                if self.split == 'train':
                    image_caption = generate_randomized_fiq_caption(image_captions)
                    reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_name = self.triplets[index]['target']
                    # Những ảnh có tên bắt đầu bằng '_' là ảnh được FLUX sinh ra, đặt trong thư mục .images
                    if target_name.startswith('_'):
                        target_image_path = base_path / 'fashionIQ_dataset' / '.images' / f"{target_name}.png"
                    else:
                        target_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{target_name}.png"
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return reference_image, target_image, image_caption

                elif self.split == 'val':
                    image_caption = generate_randomized_fiq_caption(image_captions, split='val')
                    target_name = self.triplets[index]['target']
                    return reference_name, target_name, image_caption

                elif self.split == 'test':
                    image_caption = generate_randomized_fiq_caption(image_captions, split='test')
                    reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    return reference_name, reference_image, image_caption

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                if image_name.startswith('_'):
                    image_path = base_path / 'fashionIQ_dataset' / '.images' / f"{image_name}.png"
                else:
                    image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path))
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class CIRRDataset(Dataset):
    """
       CIRR dataset class which manage CIRR data
       The dataset can be used in 'relative' or 'classic' mode:
           - In 'classic' mode the dataset yield tuples made of (image_name, image)
           - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, rel_caption) when split == train
                - (reference_name, target_name, rel_caption, group_members) when split == val
                - (pair_id, reference_name, rel_caption, group_members) when split == test1
    """

    def __init__(self, split: str, mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                  - In 'classic' mode the dataset yield tuples made of (image_name, image)
                  - In 'relative' mode the dataset yield tuples made of:
                        - (reference_image, target_image, rel_caption) when split == train
                        - (reference_name, target_name, rel_caption, group_members) when split == val
                        - (pair_id, reference_name, rel_caption, group_members) when split == test1
        :param preprocess: function which preprocesses the image
        """
        self.preprocess = preprocess
        self.mode = mode
        self.split = split

        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(base_path / 'cirr_dataset' / 'cirr' / 'captions' / f'cap.rc2.{split}.json') as f:
            self.triplets = json.load(f)

        # get a mapping from image name to relative path
        with open(base_path / 'cirr_dataset' / 'cirr' / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                rel_caption = self.triplets[index]['caption']

                if self.split == 'train':
                    reference_image_path = base_path / 'cirr_dataset' / self.name_to_relpath[reference_name]
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = base_path / 'cirr_dataset' / self.name_to_relpath[target_hard_name]
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return reference_image, target_image, rel_caption

                elif self.split == 'val':
                    target_hard_name = self.triplets[index]['target_hard']
                    return reference_name, target_hard_name, rel_caption, group_members

                elif self.split == 'test1':
                    pair_id = self.triplets[index]['pairid']
                    return pair_id, reference_name, rel_caption, group_members

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = base_path / 'cirr_dataset' / self.name_to_relpath[image_name]
                im = PIL.Image.open(image_path)
                image = self.preprocess(im)
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
