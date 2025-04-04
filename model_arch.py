import torch
import torch.nn as nn

class TamperingDetectionCNN(nn.Module):
    def __init__(self):
        super(TamperingDetectionCNN, self).__init__()
        self.features = nn.Sequential(
            # First convolution layer:
            # Input image size: 128x128, kernel_size=3, padding=0 results in 126x126 output.
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # Second convolution layer:
            # Input size 126x126 becomes 124x124.
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # Max Pooling: downsamples the feature map by 2 => 124/2 = 62 (integer division)
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        # Calculate flattened feature size:
        # 32 channels with each of size 62x62.
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
        return x  # CrossEntropyLoss applies softmax internally during training/inference.
