import torch
import json
import csv

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / len(loader), 100. * correct / total

def train_epoch_aux(model, real_loader, aux_loader, 
                    optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    aux_iter = iter(aux_loader)
    for real_inputs, real_targets in real_loader:
        try:
            aux_inputs, aux_targets = next(aux_iter)
        except StopIteration:
            aux_iter = iter(aux_loader)
            aux_inputs, aux_targets = next(aux_iter)
        inputs = torch.cat([real_inputs, aux_inputs], 
                           dim=0).to(device)
        targets = torch.cat([real_targets, aux_targets], 
                            dim=0).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / len(real_loader), 100.*correct/total

def evaluate(model, loader, device, eps=8/255):
    model.eval()
    correct, certified, total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predicted, cert = model.certify(inputs, eps)
            correct += predicted.eq(targets).sum().item()
            certified += (predicted.eq(targets) & cert).sum().item()
            total += targets.size(0)
    return 100.*correct/total, 100.*certified/total

def save_metrics(metrics, path):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved: {path}")

def save_training_log(history, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 
                        'train_acc', 'clean_acc', 'cert_acc'])
        for i in range(len(history['train_loss'])):
            writer.writerow([
                (i+1)*10,
                round(history['train_loss'][i], 4),
                round(history['train_acc'][i], 2),
                round(history['clean_acc'][i], 2),
                round(history['cert_acc'][i], 2)])
    print(f"Saved: {path}")
