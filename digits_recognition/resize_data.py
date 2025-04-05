from PIL import Image
import os

input_folder = "my_data"
output_folder = "data3"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("L")     # Konwersja do skali szarości
        img_resized = img.resize((28, 28))          # Skalowanie do 28x28
        img_resized.save(os.path.join(output_folder, filename))

print("Obrazy zostały przeskalowane do 28x28.")
