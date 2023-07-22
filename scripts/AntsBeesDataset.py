import os
from torch.utils.data import Dataset
from PIL import Image

class AntsBeesDataloader(Dataset):
    def __init__(self, root_dir, img_dir, label_dir):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_names = os.listdir(os.path.join(root_dir, img_dir))
        self.label_names = os.listdir(os.path.join(root_dir, label_dir))        

    def __getitem__(self, index):
        img_name = self.img_names[index]
        label_name = self.label_names[index]
        img_item = Image.open(os.path.join(self.root_dir, self.img_dir, img_name))
        with open(os.path.join(self.root_dir, self.label_dir, label_name), 'r') as f:
            label_item = f.readline()
        return img_item, label_item
    
    def __len__(self):
        return len(self.img_names)
    

if __name__ == "__main__":
    root_dir = f"refactor/train"
    img_dir = f'image'
    label_dir = f'label'
    dataset = AntsBeesDataloader(root_dir, img_dir, label_dir)
    print(dataset[0])
    print(len(dataset))