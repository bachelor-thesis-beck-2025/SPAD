import os
import glob
from typing import List

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FolderDataset(Dataset):
    def __init__(self, folder_path: str, transform, extensions: List[str] = None):
        super().__init__()
        self.folder_path = folder_path
        self.transform = transform
        self.extensions = extensions or ['.jpg', '.jpeg', '.png', '.bmp']
        self.image_paths = self._gather_images(folder_path, self.extensions)

    def _gather_images(self, folder_path: str, extensions: List[str]) -> List[str]:
        image_paths: List[str] = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, f'**/*{ext}'), recursive=True))
        image_paths = sorted(list({os.path.normpath(p) for p in image_paths}))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = Image.open(path).convert('RGB')
        tensor = self.transform(image)
        sample = {
            'data': tensor,
            'label': torch.tensor(0, dtype=torch.long),  # dummy label, not used in energy mode
            'image_name': path,
        }
        return sample



