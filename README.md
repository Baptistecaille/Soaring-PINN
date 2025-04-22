# Physics-Informed Neural Network (PINN) for Atmospheric Convection Modeling

A PyTorch implementation of Physics-Informed Neural Networks for simulating thermal updrafts and deep atmospheric convection, integrating fluid dynamics, thermodynamics, and microphysics.

![Example Output](https://via.placeholder.com/800x400?text=Sample+Convection+Profile)  <!-- Replace with actual image -->

## Features

- **Physics-Constrained Learning**: Enforces fundamental atmospheric equations through automatic differentiation
- **Multi-Physics Integration**:
  - Boussinesq momentum equations
  - Moist thermodynamics with latent heat release
  - Mass-flux parameterization for convective plumes
  - Cloud microphysics (condensation processes)
- **High Performance**: GPU-accelerated via PyTorch CUDA support

## Physics Formulation

The model solves these coupled equations:

| Equation | Formulation |
|----------|-------------|
| Momentum | $\frac{\partial w}{\partial t} + u\cdot\nabla w = -\frac{1}{\rho_0}\frac{\partial p'}{\partial z} + g\frac{\theta_v'}{\theta_0} + \nu_t \nabla^2 w$ |
| Heat Transport | $\frac{\partial \theta}{\partial t} + u\cdot\nabla\theta = \kappa_t \nabla^2\theta + \frac{L_v}{c_p\Pi}C$ |
| Moisture | $\frac{\partial q_v}{\partial t} + u\cdot\nabla q_v = -C + E$ |
| Mass-Flux | $\frac{\partial M_u}{\partial z} = \epsilon M_u - \delta M_u$ |

Where $C = \max(0, q_v - q_{sat})$ represents condensation rate.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Soaring-PINN.git
   cd Soaring-PINN
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the PINN model with default settings:

```bash
python train_pinn.py
```

Customize training with command-line arguments:

```bash
python train_pinn.py --hidden_layers 64 128 128 64 --epochs 5000 --lr 0.001 --n_points 10000 --initial_condition thermal_bubble
```

Available options:
- `--hidden_layers`: Size of hidden layers in the neural network
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--n_points`: Number of collocation points
- `--physics_weight`: Weight for physics-based loss
- `--data_weight`: Weight for data-based loss
- `--output_dir`: Directory to save results
- `--initial_condition`: Initial condition type (thermal_bubble, cold_pool, or random)
- `--device`: Device to use for training (cpu or cuda)

### Visualizing Results

After training, visualize the results:

```bash
python visualize_results.py --plot_type vertical_profile
```

Visualization options:
- `--plot_type`: Type of visualization (vertical_profile, 2d_slice, time_evolution)
- `--time`: Time point for visualization in seconds
- `--height`: Height for 2D horizontal slice in meters
- `--x_position`: X-position for vertical slice in meters

## Model Architecture

The PINN model consists of:

1. **Neural Network**: A fully-connected network that maps (t, x, y, z) coordinates to physical variables (w, θ, q_v, p', q_l)
2. **Automatic Differentiation**: Computes spatial and temporal derivatives for PDE constraints
3. **Physics-Based Loss**: Enforces conservation laws and physical relationships
4. **Data-Fitting Loss**: Incorporates initial/boundary conditions and observational data

## Examples

### Thermal Bubble Simulation

```bash
python train_pinn.py --initial_condition thermal_bubble --epochs 2000
python visualize_results.py --plot_type time_evolution
```

### Cold Pool Simulation

```bash
python train_pinn.py --initial_condition cold_pool --epochs 2000
python visualize_results.py --plot_type 2d_slice --height 500
```

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

2. Stensrud, D. J. (2007). Parameterization schemes: keys to understanding numerical weather prediction models. Cambridge University Press.

## License

MIT