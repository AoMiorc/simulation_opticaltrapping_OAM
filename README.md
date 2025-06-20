# 角动量光镊项目 / Angular Momentum Light Trap Project

## 项目概述 / Project Overview

**中文：**
这是一个用于模拟光学陷阱中粒子动力学的Python项目。项目专注于拉盖尔-高斯(Laguerre-Gaussian, LG)光束的生成和粒子在光阱中的运动仿真，支持多种LG模式的单粒子和多粒子动力学模拟。

**English:**
This is a Python project for simulating particle dynamics in optical traps. The project focuses on generating Laguerre-Gaussian (LG) beams and simulating particle motion in optical traps, supporting single-particle and multi-particle dynamics simulations for various LG modes.

## 功能特性 / Features

### 中文
- **LG光束生成**：支持不同模式的拉盖尔-高斯光束生成（LG01、LG61等）
- **粒子动力学仿真**：模拟微粒在光阱中的运动轨迹
- **多粒子系统**：支持多个粒子同时仿真
- **可视化功能**：提供轨迹可视化和光场强度分布显示
- **数据导出**：支持CSV格式的轨迹数据导出
- **参数可调**：可自定义光束参数、粒子属性和仿真条件

### English
- **LG Beam Generation**: Support for generating Laguerre-Gaussian beams of different modes (LG01, LG61, etc.)
- **Particle Dynamics Simulation**: Simulate particle trajectories in optical traps
- **Multi-particle Systems**: Support for simultaneous simulation of multiple particles
- **Visualization**: Provide trajectory visualization and optical field intensity distribution display
- **Data Export**: Support CSV format trajectory data export
- **Configurable Parameters**: Customizable beam parameters, particle properties, and simulation conditions

## 项目结构 / Project Structure
```
Optical_image_python/
├── simulation/                    # 核心仿真模块 / Core simulation modules
│   ├── box.py                    # 仿真盒子定义 / Simulation box definition
│   ├── environment.py            # 环境设置 / Environment settings
│   ├── particle.py               # 粒子类定义 / Particle class definition
│   ├── trap.py                   # 光阱实现 / Optical trap implementation
│   └── visualizer.py             # 可视化工具 / Visualization tools
├── generate_lg61_beam.py          # LG61光束生成脚本 / LG61 beam generation script
├── test_lg01_single_particle.py   # LG01单粒子测试 / LG01 single particle test
├── test_lg01_two_particles.py     # LG01双粒子测试 / LG01 two particles test
├── test_lg61_single_particle.py   # LG61单粒子测试 / LG61 single particle test
└── *.csv                          # 轨迹数据文件 / Trajectory data files
```

## 安装要求 / Installation Requirements

### 中文
确保您的系统已安装以下Python包：

### English
Ensure your system has the following Python packages installed:

```bash
pip install numpy matplotlib pandas scipy

依赖包 / Dependencies:

- numpy
- matplotlib
- pandas
- scipy

```

## 可选安装方式 / Alternative Installation Methods

### requirements.txt:

echo "numpy>=1.20.0" > requirements.txt
echo "matplotlib>=3.3.0" >> requirements.txt
echo "pandas>=1.3.0" >> requirements.txt
echo "scipy>=1.7.0" >> requirements.txt

### pip:
pip install -r requirements.txt

### conda:

conda install numpy matplotlib pandas scipy


## 核心模块详解 / Core Modules Documentation

### simulation/ 目录 / simulation/ Directory

#### 1. particle.py - 粒子系统 / Particle System
**中文功能：**
- `Particle` 类：定义粒子的基本属性（位置、速度、质量、半径等）
- `ParticleFactory` 类：粒子工厂，提供预定义材料的粒子创建
  - `create_polystyrene_sphere()`: 创建聚苯乙烯球
  - `create_silica_sphere()`: 创建二氧化硅球
  - `create_gold_nanoparticle()`: 创建金纳米粒子
  - `create_batch_particles()`: 批量创建相同粒子
  - `create_random_distribution()`: 创建随机分布的粒子群

**English Functions:**
- `Particle` class: Defines basic particle properties (position, velocity, mass, radius, etc.)
- `ParticleFactory` class: Particle factory providing predefined material particle creation
  - `create_polystyrene_sphere()`: Create polystyrene spheres
  - `create_silica_sphere()`: Create silica spheres
  - `create_gold_nanoparticle()`: Create gold nanoparticles
  - `create_batch_particles()`: Batch create identical particles
  - `create_random_distribution()`: Create randomly distributed particle groups

#### 2. environment.py - 环境模拟 / Environment Simulation
**中文功能：**
- `Environment` 类：定义仿真环境参数
  - 介质类型（气体/液体）设置
  - 温度和粘度控制
  - 阻尼系数计算
  - 布朗运动噪声生成

**English Functions:**
- `Environment` class: Define simulation environment parameters
  - Medium type (gas/liquid) settings
  - Temperature and viscosity control
  - Damping coefficient calculation
  - Brownian motion noise generation

#### 3. trap.py - 光阱系统 / Optical Trap System
**中文功能：**
- `OpticalTrap` 类：光学陷阱的核心实现
  - 光场强度分布计算
  - 光学力和扭矩计算
  - 坡印廷矢量场分析
  - 角动量密度计算
  - 支持自定义光场函数

**English Functions:**
- `OpticalTrap` class: Core implementation of optical traps
  - Optical field intensity distribution calculation
  - Optical force and torque calculation
  - Poynting vector field analysis
  - Angular momentum density calculation
  - Support for custom optical field functions

#### 4. box.py - 仿真容器 / Simulation Container
**中文功能：**
- `SimulationBox` 类：仿真的主控制器
  - 时间步进积分
  - 粒子运动方程求解
  - 多粒子相互作用处理
  - 轨迹数据记录和导出

**English Functions:**
- `SimulationBox` class: Main controller for simulation
  - Time-stepping integration
  - Particle motion equation solving
  - Multi-particle interaction handling
  - Trajectory data recording and export

#### 5. visualizer.py - 可视化工具 / Visualization Tools
**中文功能：**
- `TrajectoryVisualizer` 类：轨迹数据可视化
  - 2D/3D轨迹图绘制
  - 速度和力的时间序列图
  - 光场强度背景显示
  - 统计数据分析
  - 多粒子轨迹对比

**English Functions:**
- `TrajectoryVisualizer` class: Trajectory data visualization
  - 2D/3D trajectory plotting
  - Velocity and force time series plots
  - Optical field intensity background display
  - Statistical data analysis
  - Multi-particle trajectory comparison

## 测试脚本说明 / Test Scripts Documentation

### 1. test_lg01_single_particle.py
**中文：** LG01模式单粒子测试，演示涡旋光束中粒子的轨道运动
**English:** LG01 mode single particle test, demonstrating particle orbital motion in vortex beams

### 2. test_lg01_two_particles.py
**中文：** LG01模式双粒子测试，研究粒子间相互作用和集体运动
**English:** LG01 mode two-particle test, studying inter-particle interactions and collective motion

### 3. test_lg61_single_particle.py
**中文：** LG61模式单粒子测试，展示高阶LG模式的复杂光场效应
**English:** LG61 mode single particle test, showcasing complex optical field effects of higher-order LG modes

### 4. generate_lg61_beam.py
**中文：** LG61光束生成工具，用于创建和可视化LG61模式的光场分布
**English:** LG61 beam generation tool for creating and visualizing LG61 mode optical field distributions

## 使用指南 / Usage Guide

### 快速开始 / Quick Start

#### 1. 运行预设仿真 / Run Preset Simulations
```python
# 运行LG01单粒子测试 / Run LG01 single particle test
python test_lg01_single_particle.py

# 运行LG61单粒子测试 / Run LG61 single particle test
python test_lg61_single_particle.py

# 运行LG01双粒子测试 / Run LG01 two-particles test
python test_lg01_two_particles.py
```

### 自定义仿真
#### Custom Simulation 创建自定义粒子 / Create Custom Particles
```python
# 自定义粒子属性 / Customize particle properties
particle = ParticleFactory.create_polystyrene_sphere(
    position=np.array([0, 0, 0]),
    velocity=np.array([0, 0, 0]),
    radius=1e-9,
    mass=1e-26,
    charge=1.602e-19
)

# 创建预设粒子 / Create predefined particle
particle = ParticleFactory.create_silica_sphere() # 二氧化硅球 / Silica sphere
particle = ParticleFactory.create_gold_nanoparticle() # 金纳米粒子 / Gold nanoparticle  


# 创建粒子群 / Create particle group
particles = ParticleFactory.create_batch_particles(particle, 10)
```

#### 自定义环境 / Custom Environment
```python
# 设置仿真环境 / Set up simulation environment
from simulation.environment import Environment

# 水环境 / Water environment
water_env = Environment(
    medium='liquid',
    T=298.0,  # 室温 / Room temperature
    eta=0.001  # 水的粘度 / Water viscosity
)

# 空气环境 / Air environment
air_env = Environment(
    medium='gas',
    T=298.0,
    eta=1.8e-5  # 空气粘度 / Air viscosity
)
```
#### 自定义光阱 / Custom Optical Trap
```python
from simulation.trap import OpticalTrap

# LG01光阱 / LG01 optical trap
lg01_trap = OpticalTrap(
    kappa=[1e-6, 1e-6, 1e-7],  # 阱刚度 / Trap stiffness
    center=np.array([0, 0, 0]),
    wavelength=1064e-9,  # 1064nm激光 / 1064nm laser
    laser_power=0.1,  # 100mW
    w0=2e-6,  # 束腰半径 / Beam waist
    l=1  # 轨道角动量量子数 / Orbital angular momentum
)
```

#### 创建光阱 / Create Optical Trap
```python
from simulation.trap import OpticalTrap

# LG01光阱 / LG01 optical trap
lg01_trap = OpticalTrap(
    kappa=[1e-6, 1e-6, 1e-7],  # 阱刚度 / Trap stiffness
    center=np.array([0, 0, 0]),
    wavelength=1064e-9,  # 1064nm激光 / 1064nm laser
    laser_power=0.1,  # 100mW
    w0=2e-6,  # 束腰半径 / Beam waist
    l=1  # 轨道角动量量子数 / Orbital angular momentum
)
```

#### 运行仿真 / Run Simulation
```python
from simulation.box import SimulationBox

# 创建仿真盒子 / Create simulation box
sim_box = SimulationBox(
    particles=particle,
    environment=water_env,
    optical_trap=lg01_trap
)

# 运行仿真 / Run simulation
duration = 0.01  # 10ms
trajectory = sim_box.simulate(duration)
```

#### 可视化结果 / Visualize Results
```python
from simulation.visualizer import TrajectoryVisualizer

# 创建可视化器 / Create visualizer
visualizer = TrajectoryVisualizer('trajectory_data.csv')

# 绘制2D轨迹 / Plot 2D trajectory
visualizer.plot_2d_trajectory()

# 绘制3D轨迹 / Plot 3D trajectory
visualizer.plot_3d_trajectory()

# 绘制速度时间序列 / Plot velocity time series
visualizer.plot_velocity_magnitude()
```

### 输出文件说明 / Output Files Description
CSV轨迹文件 / CSV Trajectory Files
- particle_trajectory_lg01_test.csv : LG01单粒子轨迹数据
- particle_trajectory_lg61_test.csv : LG61单粒子轨迹数据
- particle_trajectory_lg01_two_particles_test.csv : LG01双粒子轨迹数据

数据格式 / Data Format:
- Time (s): 时间 / Time
- X/Y/Z (m): 位置坐标 / Position coordinates
- Vx/Vy/Vz (m/s): 速度分量 / Velocity components
- Fx/Fy/Fz (N): 力分量 / Force components
- Torque_x/y/z (N·m): 扭矩分量 / Torque components

## 技术参数 / Technical Parameters
### 支持的LG模式 / Supported LG Modes
- LG01: l=1, p=0 (基础涡旋模式 / Basic vortex mode)
- LG61: l=6, p=1 (高阶复杂模式 / Higher-order complex mode)
- 可扩展支持其他模式 / Extensible for other modes
### 粒子材料库 / Particle Material Library
- 聚苯乙烯 (Polystyrene): 密度 1050 kg/m³, 折射率 1.59
- 二氧化硅 (Silica): 密度 2200 kg/m³, 折射率 1.46
- 金 (Gold): 密度 19300 kg/m³, 复折射率支持
### 仿真精度 / Simulation Precision
- 时间步长: 可调节 (1μs - 1ms) / Time step: Adjustable (1μs - 1ms)
- 空间分辨率: 纳米级 / Spatial resolution: Nanometer scale
- 力计算精度: 皮牛顿级 / Force calculation precision: Piconewton scale