import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class AtmosphericPINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for Atmospheric Convection Modeling
    Implements a neural network that respects physical constraints from fluid dynamics,
    thermodynamics, and cloud microphysics equations.
    """
    def __init__(self, hidden_layers=[64, 128, 128, 64], activation=nn.Tanh()):
        super(AtmosphericPINN, self).__init__()
        
        # Input: (t, x, y, z) coordinates
        self.input_dim = 4
        
        # Output: (w, θ, q_v, p', q_l)
        # w: vertical velocity
        # θ: potential temperature
        # q_v: water vapor mixing ratio
        # p': pressure perturbation
        # q_l: cloud liquid water mixing ratio
        self.output_dim = 5
        
        # Physical constants
        self.g = 9.81  # gravitational acceleration (m/s²)
        self.rho_0 = 1.0  # reference air density (kg/m³)
        self.theta_0 = 300.0  # reference potential temperature (K)
        self.nu_t = 0.1  # turbulent viscosity (m²/s)
        self.kappa_t = 0.1  # thermal diffusivity (m²/s)
        self.L_v = 2.5e6  # latent heat of vaporization (J/kg)
        self.c_p = 1005.0  # specific heat at constant pressure (J/kg/K)
        self.R_d = 287.0  # gas constant for dry air (J/kg/K)
        self.C_epsilon = 0.09  # TKE dissipation constant
        self.C_k = 0.1  # Turbulent viscosity constant
        
        # Neural network architecture
        layers = []
        prev_dim = self.input_dim
        
        # Hidden layers
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation)
            prev_dim = h_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 4) representing (t, x, y, z)
            
        Returns:
            Tensor of shape (batch_size, 5) representing (w, θ, q_v, p', q_l)
        """
        return self.net(x)
    
    def compute_derivatives(self, x):
        """
        Compute spatial and temporal derivatives using automatic differentiation
        
        Args:
            x: Input tensor of shape (batch_size, 4) representing (t, x, y, z)
            
        Returns:
            Dictionary containing all required derivatives for physics-based loss computation
        """
        x.requires_grad_(True)
        
        # Forward pass to get predictions
        y = self.forward(x)
        w, theta, q_v, p_prime, q_l = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4], y[:, 4:5]
        
        # Compute first-order derivatives
        dw_dt = torch.autograd.grad(w, x, torch.ones_like(w), create_graph=True)[0][:, 0:1]
        dw_dx = torch.autograd.grad(w, x, torch.ones_like(w), create_graph=True)[0][:, 1:2]
        dw_dy = torch.autograd.grad(w, x, torch.ones_like(w), create_graph=True)[0][:, 2:3]
        dw_dz = torch.autograd.grad(w, x, torch.ones_like(w), create_graph=True)[0][:, 3:4]
        
        dtheta_dt = torch.autograd.grad(theta, x, torch.ones_like(theta), create_graph=True)[0][:, 0:1]
        dtheta_dx = torch.autograd.grad(theta, x, torch.ones_like(theta), create_graph=True)[0][:, 1:2]
        dtheta_dy = torch.autograd.grad(theta, x, torch.ones_like(theta), create_graph=True)[0][:, 2:3]
        dtheta_dz = torch.autograd.grad(theta, x, torch.ones_like(theta), create_graph=True)[0][:, 3:4]
        
        dq_v_dt = torch.autograd.grad(q_v, x, torch.ones_like(q_v), create_graph=True)[0][:, 0:1]
        dq_v_dx = torch.autograd.grad(q_v, x, torch.ones_like(q_v), create_graph=True)[0][:, 1:2]
        dq_v_dy = torch.autograd.grad(q_v, x, torch.ones_like(q_v), create_graph=True)[0][:, 2:3]
        dq_v_dz = torch.autograd.grad(q_v, x, torch.ones_like(q_v), create_graph=True)[0][:, 3:4]
        
        dp_prime_dz = torch.autograd.grad(p_prime, x, torch.ones_like(p_prime), create_graph=True)[0][:, 3:4]
        
        # Compute second-order derivatives for Laplacian terms
        d2w_dx2 = torch.autograd.grad(dw_dx, x, torch.ones_like(dw_dx), create_graph=True)[0][:, 1:2]
        d2w_dy2 = torch.autograd.grad(dw_dy, x, torch.ones_like(dw_dy), create_graph=True)[0][:, 2:3]
        d2w_dz2 = torch.autograd.grad(dw_dz, x, torch.ones_like(dw_dz), create_graph=True)[0][:, 3:4]
        
        d2theta_dx2 = torch.autograd.grad(dtheta_dx, x, torch.ones_like(dtheta_dx), create_graph=True)[0][:, 1:2]
        d2theta_dy2 = torch.autograd.grad(dtheta_dy, x, torch.ones_like(dtheta_dy), create_graph=True)[0][:, 2:3]
        d2theta_dz2 = torch.autograd.grad(dtheta_dz, x, torch.ones_like(dtheta_dz), create_graph=True)[0][:, 3:4]
        
        # Compute Laplacians
        laplacian_w = d2w_dx2 + d2w_dy2 + d2w_dz2
        laplacian_theta = d2theta_dx2 + d2theta_dy2 + d2theta_dz2
        
        # Compute advection terms
        u = torch.zeros_like(w)  # Placeholder for horizontal velocity u
        v = torch.zeros_like(w)  # Placeholder for horizontal velocity v
        
        advection_w = u * dw_dx + v * dw_dy + w * dw_dz
        advection_theta = u * dtheta_dx + v * dtheta_dy + w * dtheta_dz
        advection_q_v = u * dq_v_dx + v * dq_v_dy + w * dq_v_dz
        
        # Return all derivatives in a dictionary
        return {
            'w': w,
            'theta': theta,
            'q_v': q_v,
            'p_prime': p_prime,
            'q_l': q_l,
            'dw_dt': dw_dt,
            'advection_w': advection_w,
            'dp_prime_dz': dp_prime_dz,
            'laplacian_w': laplacian_w,
            'dtheta_dt': dtheta_dt,
            'advection_theta': advection_theta,
            'laplacian_theta': laplacian_theta,
            'dq_v_dt': dq_v_dt,
            'advection_q_v': advection_q_v,
        }
    
    def compute_saturation_humidity(self, T, p):
        """
        Compute saturation humidity using Tetens formula
        
        Args:
            T: Temperature in Kelvin
            p: Pressure in Pa
            
        Returns:
            q_sat: Saturation humidity (kg/kg)
        """
        # Tetens formula for saturation vapor pressure
        e_sat = 611.0 * torch.exp(17.27 * (T - 273.15) / (T - 35.86))
        
        # Convert to mixing ratio
        q_sat = 0.622 * e_sat / (p - 0.378 * e_sat)
        
        return q_sat
    
    def compute_condensation_rate(self, q_v, q_sat):
        """
        Compute condensation rate
        
        Args:
            q_v: Water vapor mixing ratio (kg/kg)
            q_sat: Saturation humidity (kg/kg)
            
        Returns:
            C: Condensation rate (kg/kg/s)
        """
        # Condensation occurs when q_v > q_sat
        C = torch.maximum(torch.zeros_like(q_v), q_v - q_sat)
        
        return C
    
    def compute_virtual_potential_temperature(self, theta, q_v, q_l):
        """
        Compute virtual potential temperature anomaly
        
        Args:
            theta: Potential temperature (K)
            q_v: Water vapor mixing ratio (kg/kg)
            q_l: Cloud liquid water mixing ratio (kg/kg)
            
        Returns:
            theta_v_prime: Virtual potential temperature anomaly (K)
        """
        # Virtual potential temperature anomaly
        theta_v_prime = theta - self.theta_0 + 0.61 * self.theta_0 * q_v - q_l
        
        return theta_v_prime
    
    def momentum_equation_loss(self, derivatives):
        """
        Compute loss based on momentum equation (Boussinesq approximation)
        ∂w/∂t + u·∇w = - (1/ρ₀) ∂p'/∂z + g (θ_v'/θ₀) + νₜ ∇²w
        """
        # Extract required derivatives
        dw_dt = derivatives['dw_dt']
        advection_w = derivatives['advection_w']
        dp_prime_dz = derivatives['dp_prime_dz']
        laplacian_w = derivatives['laplacian_w']
        
        # Compute virtual potential temperature anomaly
        theta_v_prime = self.compute_virtual_potential_temperature(
            derivatives['theta'], derivatives['q_v'], derivatives['q_l']
        )
        
        # Left-hand side of the equation
        lhs = dw_dt + advection_w
        
        # Right-hand side of the equation
        rhs = -(1.0 / self.rho_0) * dp_prime_dz + self.g * (theta_v_prime / self.theta_0) + self.nu_t * laplacian_w
        
        # Compute mean squared error
        return torch.mean((lhs - rhs) ** 2)
    
    def heat_transport_equation_loss(self, derivatives, T, p):
        """
        Compute loss based on heat transport equation
        ∂θ/∂t + u·∇θ = κₜ ∇²θ + (L_v/c_p Π) C
        """
        # Extract required derivatives
        dtheta_dt = derivatives['dtheta_dt']
        advection_theta = derivatives['advection_theta']
        laplacian_theta = derivatives['laplacian_theta']
        
        # Compute Exner function (Π)
        p0 = 1.0e5  # reference pressure (Pa)
        Pi = (p / p0) ** (self.R_d / self.c_p)
        
        # Compute saturation humidity and condensation rate
        q_sat = self.compute_saturation_humidity(T, p)
        C = self.compute_condensation_rate(derivatives['q_v'], q_sat)
        
        # Left-hand side of the equation
        lhs = dtheta_dt + advection_theta
        
        # Right-hand side of the equation
        rhs = self.kappa_t * laplacian_theta + (self.L_v / (self.c_p * Pi)) * C
        
        # Compute mean squared error
        return torch.mean((lhs - rhs) ** 2)
    
    def moisture_conservation_loss(self, derivatives):
        """
        Compute loss based on moisture conservation equation
        ∂q_v/∂t + u·∇q_v = -C + E
        """
        # Extract required derivatives
        dq_v_dt = derivatives['dq_v_dt']
        advection_q_v = derivatives['advection_q_v']
        
        # For simplicity, we'll assume no evaporation (E = 0)
        E = torch.zeros_like(dq_v_dt)
        
        # Compute saturation humidity and condensation rate
        # Note: This is a simplified approach; in practice, you would need temperature and pressure
        q_sat = torch.zeros_like(derivatives['q_v']) + 0.015  # Placeholder value
        C = self.compute_condensation_rate(derivatives['q_v'], q_sat)
        
        # Left-hand side of the equation
        lhs = dq_v_dt + advection_q_v
        
        # Right-hand side of the equation
        rhs = -C + E
        
        # Compute mean squared error
        return torch.mean((lhs - rhs) ** 2)
    
    def continuity_equation_loss(self, derivatives):
        """
        Compute loss based on continuity equation (anelastic approximation)
        ∇·(ρ₀ u) = 0
        """
        # This is a simplified implementation
        # In practice, you would need to compute divergence of the velocity field
        # For now, we'll just return a placeholder
        return torch.tensor(0.0, requires_grad=True)
    
    def physics_loss(self, x, T, p):
        """
        Compute total physics-based loss
        
        Args:
            x: Input tensor of shape (batch_size, 4) representing (t, x, y, z)
            T: Temperature in Kelvin
            p: Pressure in Pa
            
        Returns:
            Total physics-based loss
        """
        # Compute all required derivatives
        derivatives = self.compute_derivatives(x)
        
        # Compute individual loss terms
        momentum_loss = self.momentum_equation_loss(derivatives)
        heat_loss = self.heat_transport_equation_loss(derivatives, T, p)
        moisture_loss = self.moisture_conservation_loss(derivatives)
        continuity_loss = self.continuity_equation_loss(derivatives)
        
        # Combine losses with appropriate weights
        total_loss = momentum_loss + heat_loss + moisture_loss + continuity_loss
        
        return total_loss
    
    def data_loss(self, x, y_true):
        """
        Compute loss based on data (if available)
        
        Args:
            x: Input tensor of shape (batch_size, 4) representing (t, x, y, z)
            y_true: Ground truth tensor of shape (batch_size, 5)
            
        Returns:
            Data-based loss
        """
        y_pred = self.forward(x)
        return torch.mean((y_pred - y_true) ** 2)
    
    def total_loss(self, x, y_true=None, T=None, p=None, physics_weight=1.0, data_weight=1.0):
        """
        Compute total loss combining physics-based and data-based losses
        
        Args:
            x: Input tensor of shape (batch_size, 4) representing (t, x, y, z)
            y_true: Ground truth tensor of shape (batch_size, 5), optional
            T: Temperature in Kelvin
            p: Pressure in Pa
            physics_weight: Weight for physics-based loss
            data_weight: Weight for data-based loss
            
        Returns:
            Total loss
        """
        # Compute physics-based loss
        phys_loss = self.physics_loss(x, T, p) * physics_weight
        
        # Compute data-based loss if ground truth is provided
        if y_true is not None:
            data_loss = self.data_loss(x, y_true) * data_weight
            return phys_loss + data_loss
        
        return phys_loss


def train_pinn(model, collocation_points, boundary_points=None, initial_points=None, 
               data_points=None, data_values=None, T=None, p=None, 
               epochs=10000, lr=1e-3, physics_weight=1.0, data_weight=1.0):
    """
    Train the Physics-Informed Neural Network
    
    Args:
        model: AtmosphericPINN model
        collocation_points: Tensor of shape (n_points, 4) for enforcing PDE constraints
        boundary_points: Tensor of shape (n_boundary, 4) for boundary conditions
        initial_points: Tensor of shape (n_initial, 4) for initial conditions
        data_points: Tensor of shape (n_data, 4) for data fitting
        data_values: Tensor of shape (n_data, 5) for data fitting
        T: Temperature in Kelvin at collocation points
        p: Pressure in Pa at collocation points
        epochs: Number of training epochs
        lr: Learning rate
        physics_weight: Weight for physics-based loss
        data_weight: Weight for data-based loss
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5, verbose=True)
    
    # Training loop
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute loss on collocation points (PDE constraints)
        loss = model.total_loss(collocation_points, T=T, p=p, physics_weight=physics_weight)
        
        # Add data loss if available
        if data_points is not None and data_values is not None:
            data_loss = model.data_loss(data_points, data_values) * data_weight
            loss += data_loss
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        # Record loss
        loss_history.append(loss.item())
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")
    
    return loss_history


def generate_collocation_points(n_points, t_range, x_range, y_range, z_range):
    """
    Generate random collocation points for training
    
    Args:
        n_points: Number of points to generate
        t_range: Tuple (t_min, t_max) for time range
        x_range: Tuple (x_min, x_max) for x-coordinate range
        y_range: Tuple (y_min, y_max) for y-coordinate range
        z_range: Tuple (z_min, z_max) for z-coordinate range
        
    Returns:
        Tensor of shape (n_points, 4) containing random points
    """
    t = torch.rand(n_points, 1) * (t_range[1] - t_range[0]) + t_range[0]
    x = torch.rand(n_points, 1) * (x_range[1] - x_range[0]) + x_range[0]
    y = torch.rand(n_points, 1) * (y_range[1] - y_range[0]) + y_range[0]
    z = torch.rand(n_points, 1) * (z_range[1] - z_range[0]) + z_range[0]
    
    return torch.cat([t, x, y, z], dim=1)


def generate_temperature_pressure_profiles(collocation_points):
    """
    Generate temperature and pressure profiles at collocation points
    
    Args:
        collocation_points: Tensor of shape (n_points, 4)
        
    Returns:
        T: Temperature in Kelvin
        p: Pressure in Pa
    """
    # Extract z-coordinate (height)
    z = collocation_points[:, 3:4]
    
    # Standard atmosphere temperature profile (simplified)
    T0 = 288.15  # Surface temperature (K)
    lapse_rate = 0.0065  # Temperature lapse rate (K/m)
    T = T0 - lapse_rate * z
    
    # Standard atmosphere pressure profile (simplified)
    p0 = 101325.0  # Surface pressure (Pa)
    H = 8500.0  # Scale height (m)
    p = p0 * torch.exp(-z / H)
    
    return T, p


def plot_results(model, grid_points, T, p):
    """
    Plot model predictions
    
    Args:
        model: Trained AtmosphericPINN model
        grid_points: Tensor of shape (n_grid, 4) for visualization
        T: Temperature in Kelvin at grid points
        p: Pressure in Pa at grid points
    """
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass to get predictions
    with torch.no_grad():
        predictions = model(grid_points)
    
    # Extract predictions
    w = predictions[:, 0].numpy()
    theta = predictions[:, 1].numpy()
    q_v = predictions[:, 2].numpy()
    p_prime = predictions[:, 3].numpy()
    q_l = predictions[:, 4].numpy()
    
    # Extract coordinates for plotting
    z = grid_points[:, 3].numpy()
    
    # Create figure
    fig, axs = plt.subplots(1, 5, figsize=(20, 6))
    
    # Plot vertical velocity
    axs[0].plot(w, z)
    axs[0].set_xlabel('Vertical Velocity (m/s)')
    axs[0].set_ylabel('Height (m)')
    axs[0].set_title('Vertical Velocity Profile')
    
    # Plot potential temperature
    axs[1].plot(theta, z)
    axs[1].set_xlabel('Potential Temperature (K)')
    axs[1].set_title('Potential Temperature Profile')
    
    # Plot water vapor mixing ratio
    axs[2].plot(q_v, z)
    axs[2].set_xlabel('Water Vapor Mixing Ratio (kg/kg)')
    axs[2].set_title('Water Vapor Profile')
    
    # Plot pressure perturbation
    axs[3].plot(p_prime, z)
    axs[3].set_xlabel('Pressure Perturbation (Pa)')
    axs[3].set_title('Pressure Perturbation Profile')
    
    # Plot cloud liquid water
    axs[4].plot(q_l, z)
    axs[4].set_xlabel('Cloud Liquid Water (kg/kg)')
    axs[4].set_title('Cloud Liquid Water Profile')
    
    plt.tight_layout()
    plt.savefig('pinn_results.png')
    plt.show()


def main():
    """
    Main function to demonstrate the PINN model
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create PINN model
    model = AtmosphericPINN(hidden_layers=[64, 128, 128, 64])
    
    # Generate collocation points
    t_range = (0.0, 3600.0)  # 1 hour simulation
    x_range = (-5000.0, 5000.0)  # 10 km domain in x
    y_range = (-5000.0, 5000.0)  # 10 km domain in y
    z_range = (0.0, 10000.0)  # 10 km domain in z
    
    n_points = 10000
    collocation_points = generate_collocation_points(n_points, t_range, x_range, y_range, z_range)
    
    # Generate temperature and pressure profiles
    T, p = generate_temperature_pressure_profiles(collocation_points)
    
    # Train the model
    loss_history = train_pinn(model, collocation_points, T=T, p=p, epochs=1000, lr=1e-3)
    
    # Generate grid points for visualization
    z_grid = torch.linspace(0.0, 10000.0, 100).unsqueeze(1)
    t_grid = torch.zeros_like(z_grid)
    x_grid = torch.zeros_like(z_grid)
    y_grid = torch.zeros_like(z_grid)
    grid_points = torch.cat([t_grid, x_grid, y_grid, z_grid], dim=1)
    
    # Generate temperature and pressure profiles for grid points
    T_grid, p_grid = generate_temperature_pressure_profiles(grid_points)
    
    # Plot results
    plot_results(model, grid_points, T_grid, p_grid)
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.grid(True)
    plt.savefig('loss_history.png')
    plt.show()


if __name__ == "__main__":
    main()