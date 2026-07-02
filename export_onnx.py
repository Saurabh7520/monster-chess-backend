import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNMonsterNet(nn.Module):
    def __init__(self):
        super(CNNMonsterNet, self).__init__()
        # Matches the exact shapes from your master checkpoint!
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.output(x)

# Initialize, load the master backup, and set to evaluation mode
model = CNNMonsterNet()
model.load_state_dict(torch.load("chess_brain_master.pth", weights_only=True))
model.eval()

# Create dummy input that matches the 12-channel 8x8 chess matrix
dummy_input = torch.randn(1, 12, 8, 8)

# Export directly to the ONNX file your video API is waiting for
torch.onnx.export(
    model, 
    dummy_input, 
    "monster_brain.onnx", 
    export_params=True, 
    opset_version=11, 
    input_names=['input'], 
    output_names=['output']
)
print("💪 Brain successfully compiled to monster_brain.onnx!")