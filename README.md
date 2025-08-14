# Optical Tweezers Simulation Framework

## Project Overview

**English:**
This is a comprehensive Python framework for simulating optical tweezers experiments. The framework provides realistic particle dynamics simulation in optical traps, supporting various beam modes (Laguerre-Gaussian, Hermite-Gaussian), multiple particle interactions, and comprehensive visualization tools.

**中文：**
这是一个用于模拟光学镊子实验的综合Python框架。该框架提供光阱中粒子动力学的真实仿真，支持多种光束模式（拉盖尔-高斯、厄米特-高斯）、多粒子相互作用和全面的可视化工具。

## Key Features

### English
- Multi-particle Simulation : Support for single and multiple particle dynamics
- Various Beam Modes : LG (Laguerre-Gaussian) and LP (Linearly Polarized) beam modes
- Realistic Physics : Brownian motion, optical forces, torques, and angular momentum transfer
- CSV Field Import : Load experimental optical field data from CSV files
- Comprehensive Visualization : Trajectory plotting, field visualization, and video generation
- Data Export : Export simulation results to CSV format for further analysis
- Configurable Parameters : Customizable beam parameters, particle properties, and simulation conditions

### 中文
- 多粒子仿真 ：支持单粒子和多粒子动力学
- 多种光束模式 ：LG（拉盖尔-高斯）和LP（线偏振）光束模式
- 真实物理 ：布朗运动、光学力、扭矩和角动量传递
- CSV光场导入 ：从CSV文件加载实验光场数据
- 全面可视化 ：轨迹绘制、光场可视化和视频生成
- 数据导出 ：将仿真结果导出为CSV格式进行进一步分析
- 可配置参数 ：可自定义光束参数、粒子属性和仿真条件

## 项目结构 / Project Structure
```
Optical_image_python/
├── simulation/                    # Core simulation modules
│   ├── particle.py               # Particle class definitions
│   ├── environment.py            # Environment settings
│   ├── trap.py                   # Optical trap implementation
│   ├── box.py                    # Simulation box controller
│   └── visualizer.py             # Visualization tools
├── test0714/                     # Test scripts and examples
│   ├── csv_reader_singleparticle.py  # Single particle CSV example
│   └── *.csv                     # Optical field data files
└── *.py                          # Various test scripts
```

## Installation Requirements

### English
Ensure your system has the following Python packages installed:

### 中文
确保您的系统已安装以下Python包：

```bash
pip install numpy matplotlib pandas scipy

```

Dependencies:

- numpy >= 1.20.0
- matplotlib >= 3.3.0
- pandas >= 1.3.0
- scipy >= 1.7.0
## Core Modules Documentation
### 1. particle.py - Particle System
Public Classes and Methods:
 Particle Class
- __init__(mass, radius, position, refractive_index) : Initialize particle with physical properties ParticleFactory Class
- create_particle(material, radius, position) : Create particle from predefined materials
- create_polystyrene_sphere(radius, position) : Create polystyrene spherical particle
- create_silica_sphere(radius, position) : Create silica spherical particle
- create_gold_nanoparticle(radius, position) : Create gold nanoparticle
- create_multiple_particles(material, radius, positions) : Batch create identical particles
- create_random_particles(material, radius, num_particles, x_range, y_range, z_range) : Create randomly distributed particles
- add_material(name, density, refractive_index) : Add new material type
- get_available_materials() : Get list of available materials
Supported Materials: polystyrene, silica, gold, silver, latex, pmma, glass

### 2. environment.py - Environment Simulation
Public Classes and Methods:
 Environment Class
- __init__(medium, T, eta, P_gas, M_gas, refractive_index) : Initialize environment parameters
- get_drag_coefficient(particle) : Calculate translational drag coefficient using Stokes' law (liquid) or kinetic theory (gas)
- get_angular_drag_coefficient(particle) : Calculate rotational drag coefficient
Features:

- Support for both liquid and gas environments
- Automatic Knudsen number calculation for gas environments
- Temperature and viscosity control
- Pressure-dependent damping for gas media
### 3. trap.py - Optical Trap System
Public Classes and Methods:
 OpticalTrap Class
- __init__(center, wavelength, laser_power, w0, l) : Initialize optical trap parameters
- setup_csv_field(intensity_csv, phase_csv, x_range, y_range, z_range) : Load optical field from CSV files
- set_field(grid_x, grid_y, grid_z, field_function, phase_function) : Set custom optical field
- get_intensity_at_position(position) : Get normalized light intensity at specified position
- get_force(position, particle_radius, refractive_index) : Calculate optical force based on intensity gradient
- calculate_intensity_gradient(position) : Calculate intensity gradient at specified position
- calculate_poynting_field() : Calculate Poynting vector field for angular momentum analysis
- calculate_angular_momentum_field() : Calculate angular momentum density field
- calculate_angular_momentum_axis() : Calculate angular momentum along optical axis
Features:

- Support for arbitrary beam modes through CSV import
- Automatic orbital angular momentum (OAM) calculation
- Poynting vector and angular momentum density analysis
- Gradient force calculation using Rayleigh scattering approximation
### 4. box.py - Simulation Container
Public Classes and Methods:
 SimulationBox Class
- __init__(particles, environment, optical_trap, timestep) : Initialize simulation container
- simulate(duration, show_progress) : Run complete simulation with progress tracking
- get_trajectory() : Get trajectory data for all particles
- save_trajectory_to_csv(filename) : Save multi-particle trajectory data to CSV
- _step() : Perform single simulation time step (internal method)
Features:

- Multi-particle dynamics integration
- Brownian motion simulation
- Optical force and torque calculation
- Real-time progress tracking
- Comprehensive data recording (position, velocity, force, angular velocity, torque)
### 5. visualizer.py - Visualization Tools
Public Classes and Methods:
 TrajectoryVisualizer Class
- __init__(csv_file, simulation_box, optical_trap) : Initialize visualizer with data source
- load_data(csv_file) : Load trajectory data from CSV file
- load_from_box(simulation_box, csv_file) : Load data from SimulationBox object
- load_from_trap(optical_trap) : Load data from OpticalTrap object
- plot_2d_trajectory(plane, particle_ids, show_start_end, trail_length) : Plot 2D particle trajectories
- plot_3d_trajectory(particle_ids, show_start_end, trail_length) : Plot 3D particle trajectories
- plot_velocity_magnitude(particle_ids) : Plot velocity magnitude vs time
- plot_force_magnitude(particle_ids) : Plot force magnitude vs time
- plot_all_magnitudes(particle_ids) : Plot all physical quantities vs time
- plot_2d_trajectory_with_phase(plane, particle_ids, optical_trap) : Plot trajectories with phase background
- plot_2d_trajectory_with_field(plane, particle_ids, optical_trap) : Plot trajectories with intensity background
- get_statistics() : Calculate comprehensive motion statistics
- analyze_and_visualize_default(sim_box, show_plots) : Complete analysis with default configuration
- create_trajectory_video(output_filename, fps, duration_sec, trail_length, zoom_to_particle, optical_trap, plane) : Create trajectory animation video
Features:

- Multi-particle trajectory visualization
- Optical field background display
- Statistical analysis (average speed, orbital period, etc.)
- Video generation with customizable parameters
- Support for multiple projection planes (xy, xz, yz)
## Usage Examples
### Single Particle Simulation

```python
# Create particle and environment
particle = ParticleFactory.create_polystyrene_sphere(radius=50e-9)
environment = Environment(medium='liquid', T=298.0)

# Create optical trap
trap = OpticalTrap(wavelength=1064e-9, laser_power=0.1, w0=2e-6)

# Run simulation
sim_box = SimulationBox(particle, environment, trap)
trajectory = sim_box.simulate(duration=0.01)
```

Multi-Particle Simulation
```python
# Create multiple particles
particles = ParticleFactory.create_random_particles(
    material='polystyrene', 
    radius=50e-9, 
    num_particles=5,
    x_range=(-1e-6, 1e-6),
    y_range=(-1e-6, 1e-6),
    z_range=(-0.5e-6, 0.5e-6)
)

# Run multi-particle simulation
sim_box = SimulationBox(particles, environment, trap)
trajectory = sim_box.simulate(duration=0.01)
```

isualization and Analysis
```python
# Load and visualize results
visualizer = TrajectoryVisualizer('trajectory.csv')
visualizer.plot_3d_trajectory()
visualizer.plot_all_magnitudes()
stats = visualizer.get_statistics()

# Create animation video
visualizer.create_trajectory_video(
    output_filename='particle_motion.mp4',
    fps=30,
    duration_sec=10
)
```
## Output Files
### CSV Trajectory Formata
The simulation generates CSV files with the following columns:

- Particle_ID : Particle identifier
- Time (s) : Simulation time
- X (m), Y (m), Z (m) : Particle position
- Vx (m/s), Vy (m/s), Vz (m/s) : Particle velocity
- Fx (N), Fy (N), Fz (N) : Applied forces
- ωx (rad/s), ωy (rad/s), ωz (rad/s) : Angular velocity
- τx (pN·μm), τy (pN·μm), τz (pN·μm) : Applied torquesa
