import torch
import torch.nn as nn

# CNN Model
class MathSymbolCNN(nn.Module):
    def __init__(self, num_classes):
        super(MathSymbolCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Conv1
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # Conv2 (stride=2 downsamples)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Conv3
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Conv4 (stride=2 downsamples)
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # Conv5
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 768),  # Fully connected layer 1
            nn.ReLU(),
            nn.Linear(768, 128),  # Fully connected layer 2
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Output layer (num_classes = number of symbols)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
