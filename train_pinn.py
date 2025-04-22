import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pinn_model import AtmosphericPINN, generate_collocation_points, generate_temperature_pressure_profiles, train_pinn, plot_results


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Train Physics-Informed Neural Network for Atmospheric Convection')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[64, 128, 128, 64],
                        help='Size of hidden layers')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--n_points', type=int, default=10000,
                        help='Number of collocation points')
    parser.add_argument('--physics_weight', type=float, default=1.0,
                        help='Weight for physics-based loss')
    parser.add_argument('--data_weight', type=float, default=1.0,
                        help='Weight for data-based loss')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for training (cpu or cuda)')
    parser.add_argument('--initial_condition', type=str, default='thermal_bubble',
                        help='Initial condition type (thermal_bubble, cold_pool, or random)')
    
    return parser.parse_args()


def setup_domain():
    """
    Set up the computational domain
    """
    # Domain ranges
    t_range = (0.0, 3600.0)  # 1 hour simulation
    x_range = (-5000.0, 5000.0)  # 10 km domain in x
    y_range = (-5000.0, 5000.0)  # 10 km domain in y
    z_range = (0.0, 10000.0)  # 10 km domain in z
    
    return t_range, x_range, y_range, z_range


def create_initial_condition(type_name, grid_points):
    """
    Create initial condition data based on specified type
    
    Args:
        type_name: Type of initial condition ('thermal_bubble', 'cold_pool', or 'random')
        grid_points: Grid points tensor of shape (n_points, 4)
        
    Returns:
        initial_data: Tensor of shape (n_points, 5) with initial values
    """
    # Extract coordinates
    t = grid_points[:, 0:1]
    x = grid_points[:, 1:2]
    y = grid_points[:, 2:3]
    z = grid_points[:, 3:4]
    
    # Initialize data tensor
    initial_data = torch.zeros((grid_points.shape[0], 5))
    
    # Set initial values based on type
    if type_name == 'thermal_bubble':
        # Create a warm bubble in the center of the domain
        r = torch.sqrt((x/1000.0)**2 + (y/1000.0)**2 + ((z-2000.0)/1000.0)**2)
        theta_perturbation = 2.0 * torch.exp(-r**2 / 2.0)  # 2K perturbation with Gaussian profile
        
        # Set initial values
        initial_data[:, 0] = 0.0  # w (vertical velocity)
        initial_data[:, 1] = 300.0 + theta_perturbation.squeeze()  # θ (potential temperature)
        initial_data[:, 2] = 0.01 * torch.ones_like(t).squeeze()  # q_v (water vapor mixing ratio)
        initial_data[:, 3] = 0.0  # p' (pressure perturbation)
        initial_data[:, 4] = 0.0  # q_l (cloud liquid water)
        
    elif type_name == 'cold_pool':
        # Create a cold pool near the surface
        r = torch.sqrt((x/1000.0)**2 + (y/1000.0)**2 + ((z-500.0)/500.0)**2)
        theta_perturbation = -3.0 * torch.exp(-r**2 / 2.0)  # -3K perturbation with Gaussian profile
        
        # Set initial values
        initial_data[:, 0] = 0.0  # w (vertical velocity)
        initial_data[:, 1] = 300.0 + theta_perturbation.squeeze()  # θ (potential temperature)
        initial_data[:, 2] = 0.012 * torch.ones_like(t).squeeze()  # q_v (water vapor mixing ratio)
        initial_data[:, 3] = 0.0  # p' (pressure perturbation)
        initial_data[:, 4] = 0.0  # q_l (cloud liquid water)
        
    else:  # random
        # Random initial conditions
        initial_data[:, 0] = 0.1 * torch.randn_like(t).squeeze()  # w (vertical velocity)
        initial_data[:, 1] = 300.0 + 0.5 * torch.randn_like(t).squeeze()  # θ (potential temperature)
        initial_data[:, 2] = 0.01 + 0.002 * torch.randn_like(t).squeeze()  # q_v (water vapor mixing ratio)
        initial_data[:, 3] = 10.0 * torch.randn_like(t).squeeze()  # p' (pressure perturbation)
        initial_data[:, 4] = 0.0  # q_l (cloud liquid water)
    
    return initial_data


def main():
    """
    Main function to train the PINN model
    """
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up domain
    t_range, x_range, y_range, z_range = setup_domain()
    
    # Create PINN model
    model = AtmosphericPINN(hidden_layers=args.hidden_layers).to(device)
    print(f"Created PINN model with architecture: {args.hidden_layers}")
    
    # Generate collocation points
    collocation_points = generate_collocation_points(
        args.n_points, t_range, x_range, y_range, z_range
    ).to(device)
    print(f"Generated {args.n_points} collocation points")
    
    # Generate temperature and pressure profiles
    T, p = generate_temperature_pressure_profiles(collocation_points)
    print("Generated temperature and pressure profiles")
    
    # Create initial condition data
    initial_data = create_initial_condition(args.initial_condition, collocation_points)
    print(f"Created initial condition: {args.initial_condition}")
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs...")
    loss_history = train_pinn(
        model, collocation_points, 
        data_points=collocation_points, data_values=initial_data,
        T=T, p=p, 
        epochs=args.epochs, lr=args.lr,
        physics_weight=args.physics_weight, data_weight=args.data_weight
    )
    print("Training completed")
    
    # Save the trained model
    model_path = os.path.join(args.output_dir, 'pinn_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Generate grid points for visualization
    z_grid = torch.linspace(0.0, 10000.0, 100).unsqueeze(1).to(device)
    t_grid = torch.zeros_like(z_grid)
    x_grid = torch.zeros_like(z_grid)
    y_grid = torch.zeros_like(z_grid)
    grid_points = torch.cat([t_grid, x_grid, y_grid, z_grid], dim=1)
    
    # Generate temperature and pressure profiles for grid points
    T_grid, p_grid = generate_temperature_pressure_profiles(grid_points)
    
    # Plot results
    plot_results(model, grid_points, T_grid, p_grid)
    plt.savefig(os.path.join(args.output_dir, 'pinn_results.png'))
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'loss_history.png'))
    
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()