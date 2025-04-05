import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from model import NeuralNet
import torchvision.transforms as transforms

# Transformacje: konwersja obrazów do tensora i normalizacja
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Pobranie zbioru EMNIST MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Tworzenie modelu, funkcji kosztu i optymalizatora
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # urządzenie do trenowania
model = NeuralNet().to(device)
criterion = nn.CrossEntropyLoss() # mierzy różnicę między prawdziwymi etykietami a przewidywaniami modelu
optimizer = optim.Adam(model.parameters(), lr=0.001) # dostosowuje wagi modelu, aby minimalizować funkcję kosztu

# Trenowanie modelu
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()               # zerowanie wartości gradientu
        outputs = model(images)             # przekazywanie w przód
        loss = criterion(outputs, labels)   # oblicza stratę na podstawie przewidywań i prawdziwych etykiet
        loss.backward()                     # propagacja wstecznej błędu
        optimizer.step()                    # aktualizacja wagi sieci na podstawie obliczonego gradientu
        running_loss += loss.item()         # obliczenie średniej pomyłki obecnej iteracji treningowej
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Zapisanie modelu
torch.save(model.state_dict(), "model3.pth")
print("Model zapisany jako model.pth")