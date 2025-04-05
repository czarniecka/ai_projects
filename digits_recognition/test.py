import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import NeuralNet
from sklearn.metrics import classification_report

# Transformacje: konwersja obrazów do tensora i normalizacja
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Pobranie zbioru testowego
folder_path = 'data2'  
test_dataset = CustomDataset(folder_path=folder_path, transform=transform)
#test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Wczytanie modelu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet().to(device)
model.load_state_dict(torch.load("model.pth"))

# Ewaluacja modelu
model.eval()
y_true, y_pred = [], []
with torch.no_grad():                           # Nie liczymy gradientów
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)                 # Co widzi
        _, predicted = torch.max(outputs, 1)    # Wybieramy cyfrę z największą pewnością
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

for i in range(len(labels)):
    print(f"Prawdziwa cyfra: {labels[i]}, Rozpoznana jako: {predicted[i].item()}")

report = classification_report(y_true, y_pred, digits=4)
print(report)
