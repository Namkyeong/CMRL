import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):

    def __init__(self, 
                input_dim = 42, 
                hidden_dim = 42):
        super(Classifier, self).__init__()
        
        # self.classifier = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.PReLU(), nn.Linear(hidden_dim, 1))
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, embeddings):
        
        output = self.classifier(embeddings)

        return torch.sigmoid(output)