import torch
import torch.nn as nn
import os
import sys
sys.path.append('./src')

from model import LinfDistNet
from dataset import get_cifar10, get_aux_data
from utils import (train_epoch, train_epoch_aux, 
                   evaluate, save_metrics, save_training_log)
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() 
                      else 'cpu')
print(f"Device: {device}")

os.makedirs('./results', exist_ok=True)
os.makedirs('./checkpoints', exist_ok=True)

# Load CIFAR-10
train_dataset, test_dataset = get_cifar10()
train_loader = DataLoader(train_dataset, batch_size=128, 
                          shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, 
                         shuffle=False, num_workers=2)
criterion = nn.CrossEntropyLoss()

# ========== BASELINE TRAINING ==========
print("\nTraining BASELINE (no aux data)...")
print("-" * 50)
model = LinfDistNet(input_dim=3072, width=512, 
                    depth=4, num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=200)

history = {'train_loss': [], 'train_acc': [], 
           'clean_acc': [], 'cert_acc': []}

for epoch in range(200):
    loss, train_acc = train_epoch(
        model, train_loader, optimizer, criterion, device)
    scheduler.step()
    if (epoch + 1) % 10 == 0:
        clean_acc, cert_acc = evaluate(
            model, test_loader, device)
        history['train_loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['clean_acc'].append(clean_acc)
        history['cert_acc'].append(cert_acc)
        print(f"Epoch [{epoch+1}/200] "
              f"Clean: {clean_acc:.2f}% | "
              f"Cert: {cert_acc:.2f}%")
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'history': history},
            './checkpoints/baseline_checkpoint.pt')

save_metrics({
    'clean_accuracy': history['clean_acc'][-1],
    'certified_accuracy': history['cert_acc'][-1],
    'epochs': 200,
    'aux_data': False,
    'model': 'LinfDistNet',
    'dataset': 'CIFAR-10',
    'threat_model': 'l-inf',
    'epsilon': 0.03137
}, './results/baseline_metrics.json')

save_training_log(history, './results/training_log.csv')
print(f"Baseline Done!")
print(f"Clean: {history['clean_acc'][-1]:.2f}% | "
      f"Cert: {history['cert_acc'][-1]:.2f}%")

# ========== AUX TRAINING ==========
print("\nTraining WITH auxiliary data (70% EDM + 30% real)...")
print("-" * 50)
model_aux = LinfDistNet(input_dim=3072, width=512, 
                        depth=4, num_classes=10).to(device)
optimizer_aux = torch.optim.Adam(
    model_aux.parameters(), lr=0.001)
scheduler_aux = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_aux, T_max=200)

aux_dataset = get_aux_data('./data/edm_1m.npz')
aux_loader = DataLoader(aux_dataset, batch_size=299, 
                        shuffle=True, num_workers=2)

history_aux = {'train_loss': [], 'train_acc': [], 
               'clean_acc': [], 'cert_acc': []}

for epoch in range(200):
    loss, train_acc = train_epoch_aux(
        model_aux, train_loader, aux_loader,
        optimizer_aux, criterion, device)
    scheduler_aux.step()
    if (epoch + 1) % 10 == 0:
        clean_acc, cert_acc = evaluate(
            model_aux, test_loader, device)
        history_aux['train_loss'].append(loss)
        history_aux['train_acc'].append(train_acc)
        history_aux['clean_acc'].append(clean_acc)
        history_aux['cert_acc'].append(cert_acc)
        print(f"Epoch [{epoch+1}/200] "
              f"Clean: {clean_acc:.2f}% | "
              f"Cert: {cert_acc:.2f}%")
        torch.save({
            'epoch': epoch,
            'model_state': model_aux.state_dict(),
            'history': history_aux},
            './checkpoints/aux_checkpoint.pt')

save_metrics({
    'clean_accuracy': history_aux['clean_acc'][-1],
    'certified_accuracy': history_aux['cert_acc'][-1],
    'epochs': 200,
    'aux_data': True,
    'aux_size': 50000,
    'model': 'LinfDistNet',
    'dataset': 'CIFAR-10',
    'threat_model': 'l-inf',
    'epsilon': 0.03137
}, './results/improved_metrics.json')

print(f"Aux Done!")
print(f"Clean: {history_aux['clean_acc'][-1]:.2f}% | "
      f"Cert: {history_aux['cert_acc'][-1]:.2f}%")
print("\n" + "="*50)
print("FINAL COMPARISON")
print("="*50)
print(f"Baseline - Clean: {history['clean_acc'][-1]:.2f}%"
      f" | Cert: {history['cert_acc'][-1]:.2f}%")
print(f"With Aux - Clean: {history_aux['clean_acc'][-1]:.2f}%"
      f" | Cert: {history_aux['cert_acc'][-1]:.2f}%")
