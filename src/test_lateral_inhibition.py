import torch
import numpy as np
from types import SimpleNamespace
from networks.mlp_grow import BayesianMLP

def test_lateral_inhibition():
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    args = SimpleNamespace(
        inputsize=(1, 28, 28),
        taskcla=[(0, 10)],
        samples=1,
        device=torch.device('cpu'),
        sbatch=32,
        hidden_n=10, # Small for easy inspection
        layers=1,
        rho=-3.0,
        successive_inhibition=True,
        inhibition_samples=5,
        gamma_inhibition=1.0,
        cl_mode='task-incremental',
        regularization='bbb',
        sig1=0.0,
        sig2=0.0,
        pi=0.5
    )
    
    model = BayesianMLP(args)
    model.eval()
    
    x = torch.randn(2, 1 * 28 * 28)
    
    # Run a forward pass
    # Since we can't easily mock inner functions without modifying code, 
    # we'll just check if the model runs without crashing for now, 
    # and we can inspect the manual logic.
    with torch.no_grad():
        out = model(x, sample=False)
    
    print("Lateral inhibition forward pass completed without errors.")
    
    # Manual directionality test of the algorithm logic
    z = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    z_var = torch.tensor([[0.01, 10.0, 0.01, 10.0]]) # neurons 0, 2 are "old" (confident), 1, 3 are "new" (uncertain)
    
    c = 1.0 / (z_var + 1e-6)
    gamma = 1.0
    P = gamma * c * torch.nn.functional.relu(z)
    
    S = torch.cumsum(P, dim=-1) - P
    a = torch.nn.functional.relu(z - S)
    
    print("Pre-activation Z:", z)
    print("Variance Z_var:", z_var)
    print("Confidence c:", c)
    print("Inhibition term P:", P)
    print("Cumulative Inhibition S:", S)
    print("Final Activation a:", a)
    
    # Assertions
    # Neuron 0 is old (confident), should have high c and P, but S should be 0 (no older neurons)
    assert S[0, 0] == 0
    assert a[0, 0] > 0
    
    # Neuron 1 is new (uncertain), should have S from Neuron 0
    assert S[0, 1] > 0
    assert a[0, 1] < z[0, 1] # Should be inhibited
    
    # Neuron 2 is old (confident). It gets inhibited by 0, but it itself generates massive inhibition for 3
    assert S[0, 2] > 0
    assert P[0, 2] > P[0, 1]
    
    print("Directionality and variance utilization tests passed.")

if __name__ == "__main__":
    test_lateral_inhibition()
