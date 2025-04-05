from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Wczytaj wszystkie obrazy i etykiety
        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                label = int(filename[0])
                image_path = os.path.join(folder_path, filename)
                self.images.append(image_path)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert('L')  # Konwertuj na szarość
        
        # Zastosuj transformację ToTensor przed jakimikolwiek operacjami na tensorze
        if self.transform:
            img = self.transform(img)
        
        return img, label
