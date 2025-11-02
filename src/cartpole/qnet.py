import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)
    

def predict(state: torch.Tensor, model: DQN) -> torch.Tensor:
    """Given a state, predict Q-values for all actions using the DQN model."""
    with torch.no_grad():
        q_values = model(state)
    return q_values

def rebuild_model(model_path: str, in_dim: int=4, out_dim: int=2) -> DQN:
    """Load the model weights from the specified path into the DQN model."""
    model = DQN(in_dim, out_dim) 
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model
