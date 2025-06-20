import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加simulation模块到路径 / Add simulation module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

from particle import ParticleFactory
from environment import Environment
from trap import OpticalTrap
from box import SimulationBox
from visualizer import TrajectoryVisualizer

def create_lg61_field(r, phi, z, w0=2e-6, wavelength=1064e-9, l=6, p=1):
    """
    创建LG61光束强度分布 (l=6, p=1) / Create LG61 beam intensity distribution (l=6, p=1)
    
    参数 / Parameters:
    r, phi, z: 柱坐标系坐标 / Cylindrical coordinates
    w0: 束腰半径 (beam waist) / Beam waist radius
    wavelength: 波长 / Wavelength
    l: 轨道角动量量子数 (azimuthal index) = 6 / Orbital angular momentum quantum number (azimuthal index) = 6
    p: 径向量子数 (radial index) = 1 / Radial quantum number (radial index) = 1
    """
    # 避免除零错误 / Avoid division by zero
    r = np.maximum(r, 1e-12)
    
    # 瑞利长度 / Rayleigh length
    z_R = np.pi * w0**2 / wavelength
    
    # 束腰随z变化 / Beam waist variation with z
    w_z = w0 * np.sqrt(1 + (z / z_R)**2)
    
    # 归一化径向坐标 / Normalized radial coordinate
    rho_squared = 2 * (r**2) / (w_z**2)
    
    # LG61模式的径向部分：r^6 * L_1^6(2r²/w²) * exp(-r²/w²) / Radial part of LG61 mode: r^6 * L_1^6(2r²/w²) * exp(-r²/w²)
    
    # 径向项 r^|l| = r^6 / Radial term r^|l| = r^6
    radial_power = (r / w_z) ** abs(l)
    
    # 广义拉盖尔多项式 L_1^6(ρ²) = 1 + 6 - ρ² = 7 - ρ² / Generalized Laguerre polynomial L_1^6(ρ²) = 1 + 6 - ρ² = 7 - ρ²
    laguerre_term = 7 - rho_squared
    
    # 高斯包络 / Gaussian envelope
    gaussian_envelope = np.exp(-rho_squared / 2)
    
    # 角向相位（产生涡旋）/ Angular phase (creates vortex)
    angular_phase = np.exp(1j * l * phi)
    
    # 完整的LG场振幅 / Complete LG field amplitude
    amplitude = radial_power * laguerre_term * gaussian_envelope * angular_phase
    
    # 强度 I = |E|² / Intensity I = |E|²
    intensity = np.abs(amplitude)**2
    
    # 归一化强度 / Normalize intensity
    if np.max(intensity) > 0:
        intensity = intensity / np.max(intensity)
    
    return intensity

def test_lg61_single_particle():
    """测试LG61光束中的单粒子动力学 / Test single particle dynamics in LG61 beam"""
    print("=== LG61 Beam Single Particle Optical Trap Test / LG61光束单粒子光阱测试 ===")
    
    # 1. 创建粒子 / Create particle
    particle = ParticleFactory.create_silica_sphere(
        radius=500e-9,  # 500nm半径 / 500nm radius
        position=np.array([4e-6, 0.0e-6, 0.0])  # 初始位置在第一个亮环附近 / Initial position near first bright ring
    )
    # 给粒子初始向内速度 / Give particle initial inward velocity
    particle.velocity = np.array([-20e-6, 0.0, 0.0])  # 向内速度 -20μm/s / Inward velocity -20μm/s
    print(f"Particle created / 粒子已创建: radius={particle.radius*1e9:.0f}nm, mass={particle.mass*1e15:.2f}fg")
    print(f"Initial position / 初始位置: ({particle.position[0]*1e6:.1f}, {particle.position[1]*1e6:.1f}, {particle.position[2]*1e6:.1f}) μm")
    
    # 2. 创建环境 / Create environment
    environment = Environment(
        medium='liquid',
        T=300,  # 室温 / Room temperature
        eta=1e-3  # 水的粘度 / Water viscosity
    )
    print(f"Environment setup / 环境设置: {environment.medium}, T={environment.T}K")
    
    # 3. 创建LG61光阱 / Create LG61 optical trap
    optical_trap = OpticalTrap(
        kappa=[1e-6, 1e-6, 1e-7],  # 阱刚度 [N/m] / Trap stiffness [N/m]
        center=np.array([0.0, 0.0, 0.0]),
        wavelength=1064e-9,  # 1064nm激光 / 1064nm laser
        laser_power=0.2,  # 200mW
        w0=2e-6,  # 束腰半径2μm / Beam waist radius 2μm
        l=6  # 轨道角动量量子数 l=6 / Orbital angular momentum quantum number l=6
    )
    
    # 4. 设置LG61光场 / Set up LG61 optical field
    x_range = np.linspace(-8e-6, 8e-6, 80)  # ±8μm, 80点 / ±8μm, 80 points
    y_range = np.linspace(-8e-6, 8e-6, 80)
    z_range = np.linspace(-2e-6, 2e-6, 20)  # ±2μm
    
    def lg61_field_function(x, y, z):
        """LG61光场函数 / LG61 optical field function"""
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return create_lg61_field(r, phi, z, w0=optical_trap.w0, 
                                wavelength=optical_trap.wavelength, l=6, p=1)
    
    def lg61_phase_function(x, y, z):
        """LG61相位函数 / LG61 phase function"""
        phi = np.arctan2(y, x)
        return optical_trap.l * phi  # l=6的相位 / Phase for l=6
    
    # 设置光场 / Set optical field
    optical_trap.set_field(x_range, y_range, z_range, 
                          lg61_field_function, lg61_phase_function)
    print("LG61 field setup completed / LG61光场设置完成")
    
    # 5. 创建模拟盒子 / Create simulation box
    sim_box = SimulationBox(
        particles=particle,
        environment=environment,
        optical_trap=optical_trap
    )
    
    # 设置仿真参数 / Set simulation parameters
    sim_box.timestep = 1e-6  # 1μs时间步长 / 1μs time step
    sim_box.time = 0.0
    
    # 初始化阻尼系数 / Initialize damping coefficient
    sim_box.gamma = np.array([
        environment.get_drag_coefficient(particle),
        environment.get_drag_coefficient(particle),
        environment.get_drag_coefficient(particle)
    ])
    
    print(f"Damping coefficient / 阻尼系数: γ={sim_box.gamma[0]:.2e} kg/s")
    
    # 6. 运行模拟 / Run simulation
    print("Starting simulation... / 开始模拟...")
    duration = 0.015  # 15ms模拟时间 / 15ms simulation time
    trajectory = sim_box.simulate(duration)
    
    print(f"Simulation completed / 模拟完成, total time / 总时间: {duration*1000}ms")
    print(f"Time steps / 时间步数: {len(trajectory[0]['time'])}")
    
    # 7. 保存轨迹数据 / Save trajectory data
    import pandas as pd
    
    # 转换为DataFrame并保存 - 使用可视化器期望的列名格式 / Convert to DataFrame and save - using column names expected by visualizer
    df = pd.DataFrame({
        'Time (s)': trajectory[0]['time'],
        'X (m)': trajectory[0]['position'][:, 0],
        'Y (m)': trajectory[0]['position'][:, 1],
        'Z (m)': trajectory[0]['position'][:, 2],
        'Vx (m/s)': trajectory[0]['velocity'][:, 0],
        'Vy (m/s)': trajectory[0]['velocity'][:, 1],
        'Vz (m/s)': trajectory[0]['velocity'][:, 2]
    })
    
    df.to_csv('particle_trajectory_lg61_test.csv', index=False)
    print("Trajectory data saved to particle_trajectory_lg61_test.csv / 轨迹数据已保存至 particle_trajectory_lg61_test.csv")
    
    return trajectory

if __name__ == "__main__":
    # 运行LG61测试 / Run LG61 test
    trajectory = test_lg61_single_particle()
    
    # 创建可视化器并加载CSV数据 / Create visualizer and load CSV data
    visualizer = TrajectoryVisualizer()
    visualizer.load_data('particle_trajectory_lg61_test.csv')  # 加载刚生成的CSV文件 / Load the just generated CSV file
    
    # 重新创建光阱用于可视化 / Recreate optical trap for visualization
    optical_trap = OpticalTrap(
        kappa=[1e-6, 1e-6, 1e-7],
        center=np.array([0.0, 0.0, 0.0]),
        wavelength=1064e-9,
        laser_power=0.2,
        w0=2e-6,
        l=6
    )
    
    # 重新设置光场用于可视化 / Reset optical field for visualization
    x_range = np.linspace(-8e-6, 8e-6, 60)  # 降低分辨率用于可视化 / Reduce resolution for visualization
    y_range = np.linspace(-8e-6, 8e-6, 60)
    z_range = np.linspace(-2e-6, 2e-6, 20)
    
    def lg61_field_function(x, y, z):
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return create_lg61_field(r, phi, z, w0=optical_trap.w0, 
                                wavelength=optical_trap.wavelength, l=6, p=1)
    
    def lg61_phase_function(x, y, z):
        phi = np.arctan2(y, x)
        return optical_trap.l * phi
    
    optical_trap.set_field(x_range, y_range, z_range, 
                          lg61_field_function, lg61_phase_function)
    
    # 绘制带有光场背景的轨迹图 / Plot trajectory with optical field background
    visualizer.plot_2d_trajectory_with_field(
        plane='xy',
        figsize=(12, 10),
        optical_trap=optical_trap,
        field_alpha=0.6,
        field_levels=20
    )
    
    
    print("\n=== LG61 Test Completed / LG61测试完成 ===")
    print("Results saved / 结果已保存:")
    print("- Trajectory data / 轨迹数据: particle_trajectory_lg61_test.csv")
    print("- Visualization / 可视化: lg61_single_particle_test.png")