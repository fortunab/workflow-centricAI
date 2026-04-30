import torch
import torch.nn as nn

class ProjectNormalize(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, x):
        return self.norm(self.proj(x))

class Router(nn.Module):
    def __init__(self, dim, num_routes=4):
        super().__init__()
        self.fc = nn.Linear(dim, num_routes)
    def forward(self, x):
        return torch.softmax(self.fc(x.mean(dim=1)), dim=-1)

class LatentExperts(nn.Module):
    def __init__(self, dim, num_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)) for _ in range(num_experts)])
    def forward(self, x, route_weights):
        outputs = []
        for i, expert in enumerate(self.experts):
            outputs.append(expert(x) * route_weights[:, i].view(-1, 1, 1))
        return torch.stack(outputs, dim=0).sum(dim=0)

class Reasoner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True)
    def forward(self, x):
        return self.block(x)

class Reflector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
    def forward(self, x):
        return self.linear(x)

class RULIxSkeleton(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, out_dim=512):
        super().__init__()
        self.project = ProjectNormalize(input_dim, hidden_dim)
        self.router = Router(hidden_dim)
        self.experts = LatentExperts(hidden_dim)
        self.reasoner = Reasoner(hidden_dim)
        self.reflector = Reflector(hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = self.project(x)
        weights = self.router(x)
        x = self.experts(x, weights)
        x = self.reasoner(x)
        x = self.reflector(x)
        return self.head(x), weights
