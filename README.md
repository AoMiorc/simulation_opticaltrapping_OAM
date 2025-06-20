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