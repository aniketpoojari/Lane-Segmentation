import os
from PIL import Image
import numpy as np
from random import random
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torch.utils.data import DataLoader


# DEFINE CUSTOM CLASS
class LaneDataset(Dataset):
    def __init__(
        self,
        images_dir=None,
        masks_dir=None,
        resize=128,
        training_type=None,
    ):
        # ALL IMAGE NAMES
        self.image_names = os.listdir(images_dir)

        self.training_type = training_type
        self.resize = resize

        self.transform = Compose(
            [
                Resize((self.resize, self.resize)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # GET IMAGE AND MASK PATHS
        self.images_paths = []
        self.masks_paths = []
        for image_name in self.image_names:
            self.images_paths.append(os.path.join(images_dir, image_name))
            if self.training_type in ["train", "val"]:
                self.masks_paths.append(
                    os.path.join(masks_dir, image_name.split(".")[0] + ".png")
                )

    def __getitem__(self, i):
        if self.training_type == "train" or self.training_type == "val":
            image = Image.open(self.images_paths[i])
            mask = Image.open(self.masks_paths[i])

            # GETTING LANE MASK
            mask = np.array(mask)
            mask = mask == 3
            mask = Image.fromarray(mask)

            # DATA AUGMENTATION - random horizontal flip
            if random() < 0.5:
                image, mask = F.hflip(image), F.hflip(mask)

            ## APPLY TRANSFORM
            image = self.transform(image)
            mask = Compose([Resize((self.resize, self.resize)), ToTensor()])(mask)

            return image, mask

        else:
            image = Image.open(self.images_paths[i])

            # APPLY TRANSFORM
            image = self.transform(image)

            return image

    def __len__(self):
        return len(self.image_names)


def get_data_loader(X_dir, y_dir, resize, batch_size, num_workers):

    dataset = LaneDataset(
        images_dir=X_dir,
        masks_dir=y_dir,
        resize=resize,
        training_type="train",
    )

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    return loader
