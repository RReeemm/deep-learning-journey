from torch.utils.data import Dataset
import os
from PIL import Image

class MyDataset(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir,label_dir)
        self.img_path = os.listdir(self.path)
        
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.path,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label
    
    def __len__(self):
        return len(self.img_path)
    
root_dir = "PyTorch_learning\\hymenoptera_data\\train"
ant_label_dir = "ants"
bee_label_dir = "bees"
ant_dataset = MyDataset(root_dir,ant_label_dir)
bee_dataset = MyDataset(root_dir,bee_label_dir)
img , label = ant_dataset[1]
img.show()