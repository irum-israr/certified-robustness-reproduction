import torch
import sys
sys.path.append('./src')
from model import LinfDistNet
from dataset import get_cifar10
from utils import evaluate
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() 
                      else 'cpu')
print(f"Device: {device}")

# Load model
model = LinfDistNet(input_dim=3072, width=512, 
                    depth=4, num_classes=10).to(device)
checkpoint = torch.load(
    './checkpoints/baseline_checkpoint.pt',
    map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()
print("Model loaded successfully!")

# Evaluate
_, test_dataset = get_cifar10()
test_loader = DataLoader(test_dataset, 
                         batch_size=128, shuffle=False)
clean_acc, cert_acc = evaluate(model, test_loader, device)
print(f"Clean Accuracy:     {clean_acc:.2f}%")
print(f"Certified Accuracy: {cert_acc:.2f}%")
