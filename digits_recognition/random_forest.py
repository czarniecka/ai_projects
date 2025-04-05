import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from torchvision import datasets, transforms

# Transformacje: konwersja obrazów do tensora i normalizacja
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Pobranie danych
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Przekształcenie danych do formatu akceptowanego przez Random Forest
def dataset_to_numpy(dataset):
    images = dataset.data.numpy().reshape(len(dataset), -1)  # Spłaszczenie obrazów
    labels = dataset.targets.numpy()
    return images, labels

X_train, y_train = dataset_to_numpy(train_dataset)
X_test, y_test = dataset_to_numpy(test_dataset)

# Trenowanie Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42) # (drzewa decyzyjne, ziarno losowości)
clf.fit(X_train, y_train)

# Ewaluacja
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))
