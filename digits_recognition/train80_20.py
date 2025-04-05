import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import random_split, DataLoader
from model import NeuralNet
import torchvision.transforms as transforms

# Transformacje: konwersja obrazów do tensora i normalizacja
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Pobranie zbioru treningowego MNIST
full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Podział na zbiór treningowy (80%) i walidacyjny (20%)
train_size = int(0.8 * len(full_train_dataset))  # 48 000 obrazów
val_size = len(full_train_dataset) - train_size  # 12 000 obrazów
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# DataLoadery dla treningu i walidacji
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Tworzenie modelu, funkcji kosztu i optymalizatora
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trenowanie modelu
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    
    # Pętla treningowa
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)

    # Walidacja
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # Nie liczymy gradientów dla walidacji
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Obliczanie dokładności
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

# Zapisanie modelu
torch.save(model.state_dict(), "model2.pth")
print("Model zapisany jako model2.pth")
