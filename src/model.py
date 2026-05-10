import torch
import torch.nn as nn
import torch.nn.functional as F

class LinfDistLayer(nn.Linear):
    def forward(self, x):
        abs_sum = torch.sum(torch.abs(self.weight), dim=1, keepdim=True)
        weight = self.weight / (abs_sum.clamp(min=1.0))
        return F.linear(x, weight, self.bias)

class LinfDistNet(nn.Module):
    def __init__(self, input_dim=3072, width=512, 
                 depth=4, num_classes=10):
        super().__init__()
        layers = []
        layers.append(LinfDistLayer(input_dim, width))
        layers.append(nn.ReLU())
        for _ in range(depth - 2):
            layers.append(LinfDistLayer(width, width))
            layers.append(nn.ReLU())
        layers.append(LinfDistLayer(width, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)
    
    def certify(self, x, eps=8/255):
        with torch.no_grad():
            outputs = self.forward(x)
            top2 = torch.topk(outputs, 2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
            certified = margin > 2 * eps
            predicted = outputs.argmax(dim=1)
        return predicted, certified
