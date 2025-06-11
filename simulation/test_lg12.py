from simulation import Environment, Particle, OpticalTrap, SimulationBox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create environment (liquid)
liquid_env = Environment(
    medium='liquid',        
    T=298.0,               
    eta=0.00001              # water viscosity
)

# Create particle
particle_mass = 1000 * (4/3) * np.pi * (5e-7)**3
particle = Particle(mass=particle_mass, radius=5e-7)

# Create optical trap
optical_trap = OpticalTrap(
    kappa=[1e-4, 1e-4, 0.5e-4],
    wavelength=1064e-9,
    laser_power=0.1,    
    w0=1e-6,            # 将束腰半径从2微米减小到1微米
    l=2                  
)

# Define LG12 field distribution function
def lg12_field(x, y, z):
    # Calculate r and θ
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    w0 = 1e-6  # 这里也要相应修改束腰半径
    
    # LG12 mode (p=1, l=2)
    # Radial part with p=1
    L12 = 1 - 2*(r**2/w0**2)  # Associated Laguerre polynomial L_1^2
    amplitude = (r**2/w0**2) * L12 * np.exp(-r**2/w0**2)  # r^2 for l=2
    phase = np.exp(2j * theta)  # 2j for l=2
    lg_field = (np.abs(amplitude * phase))**2
    
    return lg_field

# Plot 2D field distribution function
def plot_radial_intensity(x_grid, y_grid, field_func):
    # Get beam waist radius
    w0 = 1e-6  # Same as in lg12_field function
    
    # Create radial coordinates
    r = np.linspace(0, 2e-6, 300)  # 0 to 2 microns, 300 points
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Calculate average intensity at each r
    intensity = np.zeros_like(r)
    for i, r_val in enumerate(r):
        # Take average over multiple angles at each r
        x = r_val * np.cos(theta)
        y = r_val * np.sin(theta)
        z = np.zeros_like(x)
        intensity[i] = np.mean(field_func(x, y, z))
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot intensity distribution
    plt.plot(r*1e6, intensity, 'b-', linewidth=2, label='Intensity Distribution')
    
    # Find maximum intensity position
    max_idx = np.argmax(intensity)
    r_max = r[max_idx]
    I_max = intensity[max_idx]
    
    # Mark important features
    plt.plot(r_max*1e6, I_max, 'ro', label=f'Max Intensity: r = {r_max*1e6:.1f} μm')
    
    # Calculate FWHM
    half_max = I_max/2
    r_half = r[intensity > half_max]
    fwhm = (r_half[-1] - r_half[0])
    
    # Add FWHM annotation
    plt.axhline(y=half_max, color='g', linestyle='--', alpha=0.5)
    plt.axvline(x=r_half[0]*1e6, color='g', linestyle='--', alpha=0.5)
    plt.axvline(x=r_half[-1]*1e6, color='g', linestyle='--', alpha=0.5)
    
    # Set labels and title
    plt.xlabel('Radial Distance r (μm)')
    plt.ylabel('Normalized Intensity I (a.u.)')
    plt.title('LG12 Mode Radial Intensity Distribution')
    
    # Add text annotation
    plt.text(0.98, 0.95, 
             f'Beam Characteristics:\n' + 
             f'Max Intensity Position: {r_max*1e6:.1f} μm\n' + 
             f'FWHM: {fwhm*1e6:.1f} μm\n' + 
             f'Beam Waist (w0): {w0*1e6:.1f} μm', 
             transform=plt.gca().transAxes,
             horizontalalignment='right',
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# Set up field
x_grid = np.linspace(-2e-6, 2e-6, 150)
y_grid = np.linspace(-2e-6, 2e-6, 150)
z_grid = np.linspace(-1e-6, 1e-6, 20)
optical_trap.set_field(x_grid, y_grid, z_grid, lg12_field)

# Create simulation system
sim_box = SimulationBox(particle, optical_trap, liquid_env)

# Set initial position and velocity
sim_box.particle.position = np.array([0.5e-6, 0.5e-6, 0])
sim_box.particle.velocity = np.array([-0.001, -0.001, 0])

# Plot 2D field distribution
plot_radial_intensity(x_grid, y_grid, lg12_field)

# Plot 3D field distribution
optical_trap.plot_3d_field(num_points=50)

# Run simulation
trajectory_data = sim_box.simulate(duration=1e-3, timestep=1e-6)

# Save trajectory
sim_box.save_trajectory_to_csv("particle_trajectory_lg12.csv")

# Create real-time animation with slower speed
sim_box.animate_trajectory(duration=5e-4, timestep=5e-7)  # 增加duration并调整timestep