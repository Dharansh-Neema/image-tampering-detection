import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from PIL import Image, ImageChops, ImageEnhance
from logger import setup_logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

np.random.seed(2)
torch.manual_seed(2)
logger = setup_logger(name="train",log_file="logs/train.log")
def convert_to_ela_image(path, quality=90):
    """
    Converts an image to its Error Level Analysis (ELA) representation.
    """
    original = Image.open(path).convert('RGB')
    temp_filename = 'temp_resaved.jpg'
    original.save(temp_filename, 'JPEG', quality=quality)
    resaved = Image.open(temp_filename)
    ela_image = ImageChops.difference(original, resaved)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    # Cleanup temporary file
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    return ela_image

def build_image_list(path_to_image, label, images):
    """
    Iterates over image files in a directory and appends image path and label.
    """
    for file in tqdm(os.listdir(path_to_image), desc=f"Building list from {path_to_image}"):
        if file.lower().endswith(('.jpg', '.jpeg')):
            file_path = os.path.join(path_to_image, file)
            try:
                if os.stat(file_path).st_size > 10000:
                    # Save line as "full_path,label"
                    images.append(f"{file_path},{label}\n")
            except Exception as e:
                logger.error(f"Error with file {file_path}: {e}")
                raise e
    logger.debug("Created a list of images and append with image path and labels")
    return images

# Custom PyTorch Dataset 
class ELADataset(Dataset):
    def __init__(self, csv_file, transform=None, quality=90):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.quality = quality

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image']
        label = int(self.data.iloc[idx]['class_label'])
        try:
            # Convert image to ELA representation
            ela_image = convert_to_ela_image(img_path, quality=self.quality)
            # Resize to 128x128
            ela_image = ela_image.resize((128, 128))
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            raise RuntimeError(f"Error processing image {img_path}: {e}")

        if self.transform:
            image = self.transform(ela_image)
        else:
            # Convert to numpy and then tensor (scaling to [0,1])
            image = np.array(ela_image).astype(np.float32) / 255.0
            # Change shape from HxWxC to CxHxW
            image = torch.tensor(image).permute(2, 0, 1)
        return image, label

# CNN Model Definition 
class TamperingDetectionCNN(nn.Module):
    def __init__(self):
        super(TamperingDetectionCNN, self).__init__()
        self.features = nn.Sequential(
            # First Conv2d: valid padding => kernel_size=3 gives (128-3+1)=126
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # Second Conv2d: (126-3+1)=124
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 124/2 = 62 (integer division)
            nn.Dropout(0.25)
        )
        # Calculate flattened feature size: 32 channels, 62x62 feature map
        self.flattened_size = 32 * 62 * 62
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # Two classes: Original and Tampered
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  

def main():
    # Paths and parameters
    custom_path_original = 'dataset/training/original/'
    custom_path_tampered = 'dataset/training/tampered/'
    dataset_csv = 'dataset.csv'
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    # Build CSV if it doesn't exist
    if not os.path.exists(dataset_csv):
        images_list = []
        images_list = build_image_list(custom_path_original, '0', images_list)
        images_list = build_image_list(custom_path_tampered, '1', images_list)
        
        with open(dataset_csv, 'w') as f:
            f.write("image,class_label\n")
            for line in images_list:
                f.write(line)
        print(f"CSV file saved to {dataset_csv}")
    
    # Define transformations for ELA image (convert to tensor and normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),  
    ])
    
    full_dataset = ELADataset(csv_file=dataset_csv, transform=transform, quality=90)
    
    # Split into training and validation sets
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=5, shuffle=True)
    
    # Create subset DataLoaders
    batch_size = 100
    train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx))
    
    # Initialize model, criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TamperingDetectionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.0005, alpha=0.9, eps=1e-08)
    
    # Training parameters
    epochs = 20
    patience = 2  # Early stopping patience
    best_val_accuracy = 0.0
    epochs_no_improve = 0

    # For plotting
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / total
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch} Validation", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss = running_loss / total
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | " +
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    model_filename = f"tampering_detection.pth"
    model_save_path = os.path.join(models_dir, model_filename)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot training history
    # fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    # ax[0].plot(train_losses, label="Training Loss", color='b')
    # ax[0].plot(val_losses, label="Validation Loss", color='r')
    # ax[0].set_title("Loss")
    # ax[0].legend(loc='best')
    # ax[1].plot(train_accuracies, label="Training Accuracy", color='b')
    # ax[1].plot(val_accuracies, label="Validation Accuracy", color='r')
    # ax[1].set_title("Accuracy")
    # ax[1].legend(loc='best')
    # plt.tight_layout()
    # plt.show()

if __name__ == '__main__':
    main()
