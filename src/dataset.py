import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def get_transforms():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2009, 0.2009, 0.2009))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2009, 0.2009, 0.2009))])
    return transform_train, transform_test

def get_cifar10(data_dir='./data'):
    transform_train, transform_test = get_transforms()
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, 
        download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, 
        download=True, transform=transform_test)
    return train_dataset, test_dataset

class AuxDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label

def get_aux_data(aux_path):
    data = np.load(aux_path)
    # Sample 5000 images per class
    indices = []
    for class_id in range(10):
        class_indices = np.where(
            data['label'] == class_id)[0][:5000]
        indices.extend(class_indices)
    indices = np.array(indices)
    aux_images = data['image'][indices]
    aux_labels = data['label'][indices]
    
    transform_aux = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2009, 0.2009, 0.2009))])
    
    return AuxDataset(aux_images, aux_labels, transform_aux)
