from glob import glob
from torch.utils.data import Dataset, DataLoader
import os.path

DOG_CAT_DATASET_PATH = "/Volumes/WTTCH/数据集/archive/PetImages"


class DogAndCatsDataSet(Dataset):
    """
    继承 DataSet
    """
    def __init__(self, root_path, size=(244, 244)):
        self._root_path = root_path
        self._size = size
        self._cat_files = [file for file in os.listdir(self._root_path + "/Cat") if
                           not file.startswith(".") and file.endswith(".jpg")]
        self._dog_files = [file for file in os.listdir(self._root_path + "/Dog") if
                           not file.startswith(".") and file.endswith(".jpg")]

    def __len__(self):
        return len(self._cat_files) + len(self._dog_files)

    def __getitem__(self, idx):
        cat_size = len(self._cat_files)
        if idx >= cat_size:
            return self._dog_files[idx - cat_size], "Dog"
        else:
            return self._cat_files[idx], "Cat"


