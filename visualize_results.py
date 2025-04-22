import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from matplotlib.cm import get_cmap
from pinn_model import AtmosphericPINN, generate_temperature_pressure_profiles


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Visualize PINN model results for Atmospheric Convection')
    parser.add_argument('--model_path', type=str, default='results/pinn_model.pt',
                        help='Path to the trained model')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[64, 128, 128, 64],
                        help='Size of hidden layers (must match the trained model)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save visualization results')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for inference (cpu or cuda)')
    parser.add_argument('--plot_type', type=str, default='vertical_profile',
                        choices=['vertical_profile', '2d_slice', 'time_evolution'],
                        help='Type of visualization to generate')
    parser.add_argument('--time', type=float, default=1800.0,
                        help='Time point for visualization (seconds)')
    parser.add_argument('--height', type=float, default=2000.0,
                        help='Height for 2D horizontal slice (meters)')
    parser.add_argument('--x_position', type=float, default=0.0,
                        help='X-position for vertical slice (meters)')
    
    return parser.parse_args()


def load_model(model_path, hidden_layers, device):
    """
    Load a trained PINN model
    
    Args:
        model_path: Path to the saved model
        hidden_layers: Size of hidden layers
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    model = AtmosphericPINN(hidden_layers=hidden_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def plot_vertical_profile(model, output_dir, time=1800.0, x_pos=0.0, y_pos=0.0, device='cpu'):
    """
    Plot vertical profiles of model variables
    
    Args:
        model: Trained PINN model
        output_dir: Directory to save results
        time: Time point for visualization (seconds)
        x_pos: X-position for the profile (meters)
        y_pos: Y-position for the profile (meters)
        device: Device for computation
    """
    # Generate vertical grid points
    z_grid = torch.linspace(0.0, 10000.0, 200).unsqueeze(1).to(device)
    t_grid = torch.ones_like(z_grid) * time
    x_grid = torch.ones_like(z_grid) * x_pos
    y_grid = torch.ones_like(z_grid) * y_pos
    grid_points = torch.cat([t_grid, x_grid, y_grid, z_grid], dim=1)
    
    # Generate temperature and pressure profiles
    T_grid, p_grid = generate_temperature_pressure_profiles(grid_points)
    
    # Get model predictions
    with torch.no_grad():
        predictions = model(grid_points)
    
    # Extract predictions
    w = predictions[:, 0].cpu().numpy()  # Vertical velocity
    theta = predictions[:, 1].cpu().numpy()  # Potential temperature
    q_v = predictions[:, 2].cpu().numpy() * 1000  # Water vapor (g/kg)
    p_prime = predictions[:, 3].cpu().numpy()  # Pressure perturbation
    q_l = predictions[:, 4].cpu().numpy() * 1000  # Cloud liquid water (g/kg)
    
    # Extract coordinates for plotting
    z = z_grid.cpu().numpy()
    
    # Create figure
    fig, axs = plt.subplots(1, 5, figsize=(20, 10))
    
    # Plot vertical velocity
    axs[0].plot(w, z, 'b-', linewidth=2)
    axs[0].set_xlabel('Vertical Velocity (m/s)', fontsize=12)
    axs[0].set_ylabel('Height (m)', fontsize=12)
    axs[0].set_title('Vertical Velocity Profile', fontsize=14)
    axs[0].grid(True)
    
    # Plot potential temperature
    axs[1].plot(theta, z, 'r-', linewidth=2)
    axs[1].set_xlabel('Potential Temperature (K)', fontsize=12)
    axs[1].set_title('Potential Temperature Profile', fontsize=14)
    axs[1].grid(True)
    
    # Plot water vapor mixing ratio
    axs[2].plot(q_v, z, 'g-', linewidth=2)
    axs[2].set_xlabel('Water Vapor Mixing Ratio (g/kg)', fontsize=12)
    axs[2].set_title('Water Vapor Profile', fontsize=14)
    axs[2].grid(True)
    
    # Plot pressure perturbation
    axs[3].plot(p_prime, z, 'm-', linewidth=2)
    axs[3].set_xlabel('Pressure Perturbation (Pa)', fontsize=12)
    axs[3].set_title('Pressure Perturbation Profile', fontsize=14)
    axs[3].grid(True)
    
    # Plot cloud liquid water
    axs[4].plot(q_l, z, 'c-', linewidth=2)
    axs[4].set_xlabel('Cloud Liquid Water (g/kg)', fontsize=12)
    axs[4].set_title('Cloud Liquid Water Profile', fontsize=14)
    axs[4].grid(True)
    
    # Add time information
    fig.suptitle(f'Vertical Profiles at Time = {time/60:.1f} min, x = {x_pos/1000:.1f} km, y = {y_pos/1000:.1f} km', 
                 fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure
    output_path = os.path.join(output_dir, f'vertical_profile_t{time:.0f}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved vertical profile to {output_path}")
    
    return fig


def plot_2d_slice(model, output_dir, time=1800.0, height=2000.0, device='cpu'):
    """
    Plot 2D horizontal slice of model variables
    
    Args:
        model: Trained PINN model
        output_dir: Directory to save results
        time: Time point for visualization (seconds)
        height: Height for the slice (meters)
        device: Device for computation
    """
    # Generate 2D grid points
    n_points = 100
    x = torch.linspace(-5000.0, 5000.0, n_points)
    y = torch.linspace(-5000.0, 5000.0, n_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Create input points
    t = torch.ones_like(X) * time
    z = torch.ones_like(X) * height
    grid_points = torch.stack([t, X, Y, z], dim=-1).reshape(-1, 4).to(device)
    
    # Generate temperature and pressure profiles
    T_grid, p_grid = generate_temperature_pressure_profiles(grid_points)
    
    # Get model predictions
    with torch.no_grad():
        predictions = model(grid_points)
    
    # Reshape predictions
    predictions = predictions.reshape(n_points, n_points, 5)
    
    # Extract predictions
    w = predictions[:, :, 0].cpu().numpy()  # Vertical velocity
    theta = predictions[:, :, 1].cpu().numpy()  # Potential temperature
    q_v = predictions[:, :, 2].cpu().numpy() * 1000  # Water vapor (g/kg)
    p_prime = predictions[:, :, 3].cpu().numpy()  # Pressure perturbation
    q_l = predictions[:, :, 4].cpu().numpy() * 1000  # Cloud liquid water (g/kg)
    
    # Create figure
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()
    
    # Plot vertical velocity
    im0 = axs[0].contourf(X.cpu().numpy()/1000, Y.cpu().numpy()/1000, w, cmap='RdBu_r', levels=20)
    axs[0].set_xlabel('X (km)', fontsize=12)
    axs[0].set_ylabel('Y (km)', fontsize=12)
    axs[0].set_title('Vertical Velocity (m/s)', fontsize=14)
    plt.colorbar(im0, ax=axs[0])
    
    # Plot potential temperature
    im1 = axs[1].contourf(X.cpu().numpy()/1000, Y.cpu().numpy()/1000, theta, cmap='plasma', levels=20)
    axs[1].set_xlabel('X (km)', fontsize=12)
    axs[1].set_ylabel('Y (km)', fontsize=12)
    axs[1].set_title('Potential Temperature (K)', fontsize=14)
    plt.colorbar(im1, ax=axs[1])
    
    # Plot water vapor mixing ratio
    im2 = axs[2].contourf(X.cpu().numpy()/1000, Y.cpu().numpy()/1000, q_v, cmap='Blues', levels=20)
    axs[2].set_xlabel('X (km)', fontsize=12)
    axs[2].set_ylabel('Y (km)', fontsize=12)
    axs[2].set_title('Water Vapor Mixing Ratio (g/kg)', fontsize=14)
    plt.colorbar(im2, ax=axs[2])
    
    # Plot pressure perturbation
    im3 = axs[3].contourf(X.cpu().numpy()/1000, Y.cpu().numpy()/1000, p_prime, cmap='RdGy_r', levels=20)
    axs[3].set_xlabel('X (km)', fontsize=12)
    axs[3].set_ylabel('Y (km)', fontsize=12)
    axs[3].set_title('Pressure Perturbation (Pa)', fontsize=14)
    plt.colorbar(im3, ax=axs[3])
    
    # Plot cloud liquid water
    im4 = axs[4].contourf(X.cpu().numpy()/1000, Y.cpu().numpy()/1000, q_l, cmap='Greys', levels=20)
    axs[4].set_xlabel('X (km)', fontsize=12)
    axs[4].set_ylabel('Y (km)', fontsize=12)
    axs[4].set_title('Cloud Liquid Water (g/kg)', fontsize=14)
    plt.colorbar(im4, ax=axs[4])
    
    # Remove the last subplot
    axs[5].axis('off')
    
    # Add time and height information
    fig.suptitle(f'Horizontal Slice at Time = {time/60:.1f} min, Height = {height/1000:.1f} km', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure
    output_path = os.path.join(output_dir, f'horizontal_slice_t{time:.0f}_z{height:.0f}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved horizontal slice to {output_path}")
    
    return fig


def plot_time_evolution(model, output_dir, x_pos=0.0, device='cpu'):
    """
    Plot time evolution of model variables along a vertical slice
    
    Args:
        model: Trained PINN model
        output_dir: Directory to save results
        x_pos: X-position for the vertical slice (meters)
        device: Device for computation
    """
    # Generate grid points
    n_time = 10
    n_z = 100
    times = torch.linspace(0.0, 3600.0, n_time)  # 0 to 60 minutes
    heights = torch.linspace(0.0, 10000.0, n_z)  # 0 to 10 km
    
    # Create meshgrid
    T, Z = torch.meshgrid(times, heights, indexing='ij')
    
    # Create input points
    x = torch.ones_like(T) * x_pos
    y = torch.zeros_like(T)
    grid_points = torch.stack([T, x, y, Z], dim=-1).reshape(-1, 4).to(device)
    
    # Generate temperature and pressure profiles
    T_grid, p_grid = generate_temperature_pressure_profiles(grid_points)
    
    # Get model predictions
    with torch.no_grad():
        predictions = model(grid_points)
    
    # Reshape predictions
    predictions = predictions.reshape(n_time, n_z, 5)
    
    # Extract predictions
    w = predictions[:, :, 0].cpu().numpy()  # Vertical velocity
    theta = predictions[:, :, 1].cpu().numpy()  # Potential temperature
    q_v = predictions[:, :, 2].cpu().numpy() * 1000  # Water vapor (g/kg)
    p_prime = predictions[:, :, 3].cpu().numpy()  # Pressure perturbation
    q_l = predictions[:, :, 4].cpu().numpy() * 1000  # Cloud liquid water (g/kg)
    
    # Create figure
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()
    
    # Plot vertical velocity
    im0 = axs[0].contourf(T.cpu().numpy()/60, Z.cpu().numpy()/1000, w.T, cmap='RdBu_r', levels=20)
    axs[0].set_xlabel('Time (min)', fontsize=12)
    axs[0].set_ylabel('Height (km)', fontsize=12)
    axs[0].set_title('Vertical Velocity (m/s)', fontsize=14)
    plt.colorbar(im0, ax=axs[0])
    
    # Plot potential temperature
    im1 = axs[1].contourf(T.cpu().numpy()/60, Z.cpu().numpy()/1000, theta.T, cmap='plasma', levels=20)
    axs[1].set_xlabel('Time (min)', fontsize=12)
    axs[1].set_ylabel('Height (km)', fontsize=12)
    axs[1].set_title('Potential Temperature (K)', fontsize=14)
    plt.colorbar(im1, ax=axs[1])
    
    # Plot water vapor mixing ratio
    im2 = axs[2].contourf(T.cpu().numpy()/60, Z.cpu().numpy()/1000, q_v.T, cmap='Blues', levels=20)
    axs[2].set_xlabel('Time (min)', fontsize=12)
    axs[2].set_ylabel('Height (km)', fontsize=12)
    axs[2].set_title('Water Vapor Mixing Ratio (g/kg)', fontsize=14)
    plt.colorbar(im2, ax=axs[2])
    
    # Plot pressure perturbation
    im3 = axs[3].contourf(T.cpu().numpy()/60, Z.cpu().numpy()/1000, p_prime.T, cmap='RdGy_r', levels=20)
    axs[3].set_xlabel('Time (min)', fontsize=12)
    axs[3].set_ylabel('Height (km)', fontsize=12)
    axs[3].set_title('Pressure Perturbation (Pa)', fontsize=14)
    plt.colorbar(im3, ax=axs[3])
    
    # Plot cloud liquid water
    im4 = axs[4].contourf(T.cpu().numpy()/60, Z.cpu().numpy()/1000, q_l.T, cmap='Greys', levels=20)
    axs[4].set_xlabel('Time (min)', fontsize=12)
    axs[4].set_ylabel('Height (km)', fontsize=12)
    axs[4].set_title('Cloud Liquid Water (g/kg)', fontsize=14)
    plt.colorbar(im4, ax=axs[4])
    
    # Remove the last subplot
    axs[5].axis('off')
    
    # Add position information
    fig.suptitle(f'Time Evolution at x = {x_pos/1000:.1f} km, y = 0 km', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure
    output_path = os.path.join(output_dir, f'time_evolution_x{x_pos:.0f}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved time evolution plot to {output_path}")
    
    return fig


def analyze_physics_constraints(model, output_dir, device='cpu'):
    """
    Analyze how well the model satisfies physics constraints
    
    Args:
        model: Trained PINN model
        output_dir: Directory to save results
        device: Device for computation
    """
    # Generate test points
    n_points = 1000
    t_range = (0.0, 3600.0)
    x_range = (-5000.0, 5000.0)
    y_range = (-5000.0, 5000.0)
    z_range = (0.0, 10000.0)
    
    # Random sampling
    t = torch.rand(n_points, 1) * (t_range[1] - t_range[0]) + t_range[0]
    x = torch.rand(n_points, 1) * (x_range[1] - x_range[0]) + x_range[0]
    y = torch.rand(n_points, 1) * (y_range[1] - y_range[0]) + y_range[0]
    z = torch.rand(n_points, 1) * (z_range[1] - z_range[0]) + z_range[0]
    
    test_points = torch.cat([t, x, y, z], dim=1).to(device)
    
    # Generate temperature and pressure profiles
    T, p = generate_temperature_pressure_profiles(test_points)
    
    # Compute physics-based losses
    model.eval()
    with torch.no_grad():
        derivatives = model.compute_derivatives(test_points)
        momentum_loss = model.momentum_equation_loss(derivatives).item()
        heat_loss = model.heat_transport_equation_loss(derivatives, T, p).item()
        moisture_loss = model.moisture_conservation_loss(derivatives).item()
        continuity_loss = model.continuity_equation_loss(derivatives).item()
    
    # Create a bar chart of the losses
    losses = [momentum_loss, heat_loss, moisture_loss, continuity_loss]
    labels = ['Momentum', 'Heat Transport', 'Moisture', 'Continuity']
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, losses)
    plt.yscale('log')
    plt.ylabel('Loss Value (log scale)')
    plt.title('Physics Constraint Violations')
    plt.grid(axis='y')
    
    # Save figure
    output_path = os.path.join(output_dir, 'physics_constraints.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved physics constraint analysis to {output_path}")
    
    # Print summary
    print("\nPhysics Constraint Analysis:")
    print(f"Momentum Equation Loss: {momentum_loss:.6e}")
    print(f"Heat Transport Equation Loss: {heat_loss:.6e}")
    print(f"Moisture Conservation Loss: {moisture_loss:.6e}")
    print(f"Continuity Equation Loss: {continuity_loss:.6e}")
    
    return losses


def main():
    """
    Main function to visualize PINN model results
    """
    # Parse command line arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the trained model
    model = load_model(args.model_path, args.hidden_layers, device)
    print(f"Loaded model from {args.model_path}")
    
    # Generate visualizations based on the specified type
    if args.plot_type == 'vertical_profile':
        plot_vertical_profile(model, args.output_dir, time=args.time, 
                             x_pos=args.x_position, device=device)
    
    elif args.plot_type == '2d_slice':
        plot_2d_slice(model, args.output_dir, time=args.time, 
                     height=args.height, device=device)
    
    elif args.plot_type == 'time_evolution':
        plot_time_evolution(model, args.output_dir, x_pos=args.x_position, device=device)
    
    # Analyze physics constraints
    analyze_physics_constraints(model, args.output_dir, device=device)
    
    print("Visualization completed successfully!")


if __name__ == "__main__":
    main()