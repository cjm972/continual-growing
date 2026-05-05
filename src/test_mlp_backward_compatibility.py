import torch
import numpy as np
from types import SimpleNamespace
from networks.mlp_grow import BayesianMLP

def test_backward_compatibility():
    # Set fixed random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Mock arguments expected by BayesianMLP
    args = SimpleNamespace(
        inputsize=(1, 28, 28),
        taskcla=[(0, 10)],
        samples=1,
        device=torch.device('cpu'),
        sbatch=32,
        hidden_n=100,
        layers=1,
        rho=-3.0,
        successive_inhibition=False, # Ensure our new feature is disabled
        cl_mode='task-incremental',
        regularization='bbb',
        sig1=0.0,
        sig2=0.0,
        pi=0.5
    )
    
    # Initialize the model
    model = BayesianMLP(args)
    model.eval() # Deterministic mu usage
    
    # Generate fixed input
    x = torch.randn(2, 1 * 28 * 28)
    
    # Record baseline outputs
    with torch.no_grad():
        baseline_outputs = model(x, sample=False)
        baseline_fc1_z = model.fc1(x, sample=False)
        baseline_fc1_x = torch.nn.functional.relu(baseline_fc1_z)
        
    print("Baseline recorded successfully.")
    
    # We will save the baseline to a file to be compared against later
    torch.save({
        'outputs': baseline_outputs,
        'fc1_z': baseline_fc1_z,
        'fc1_x': baseline_fc1_x
    }, 'baseline_mlp_outputs.pt')

if __name__ == "__main__":
    test_backward_compatibility()
