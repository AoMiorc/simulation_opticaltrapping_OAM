import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

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

def test_lg61_two_particles():
    """测试LG61光束中的双粒子动力学 / Test two particle dynamics in LG61 beam"""
    print("=== LG61 Beam Two Particles Optical Trap Test / LG61光束双粒子光阱测试 ===")
    
    # 记录测试开始时间 / Record test start time
    test_start_time = time.time()
    
    # 1. 创建两个粒子，分别放在不同的轨道上 / Create two particles on different orbits
    # 粒子1：放在内轨道（第一个亮环）/ Particle 1: on inner orbit (first bright ring)
    particle1 = ParticleFactory.create_silica_sphere(
        radius=400e-9,  # 400nm半径 / 400nm radius
        position=np.array([3.0e-6, 0.0e-6, 0.0])  # 内轨道位置 / Inner orbit position
    )
    # 给粒子1初始切向速度，使其在轨道上运动 / Give particle 1 initial tangential velocity
    particle1.velocity = np.array([0.0, 15e-6, 0.0])  # 切向速度 15μm/s / Tangential velocity 15μm/s
    
    # 粒子2：放在外轨道（第二个亮环）/ Particle 2: on outer orbit (second bright ring)
    particle2 = ParticleFactory.create_silica_sphere(
        radius=400e-9,  # 400nm半径 / 400nm radius
        position=np.array([5.5e-6, 0.0e-6, 0.0])  # 外轨道位置 / Outer orbit position
    )
    # 给粒子2初始切向速度，使其在轨道上运动 / Give particle 2 initial tangential velocity
    particle2.velocity = np.array([0.0, 12e-6, 0.0])  # 稍慢的切向速度 12μm/s / Slightly slower tangential velocity 12μm/s
    
    particles = [particle1, particle2]
    
    print(f"Particle 1 created / 粒子1已创建: radius={particle1.radius*1e9:.0f}nm, mass={particle1.mass*1e15:.2f}fg")
    print(f"Initial position 1 / 初始位置1: ({particle1.position[0]*1e6:.1f}, {particle1.position[1]*1e6:.1f}, {particle1.position[2]*1e6:.1f}) μm")
    print(f"Initial velocity 1 / 初始速度1: ({particle1.velocity[0]*1e6:.1f}, {particle1.velocity[1]*1e6:.1f}, {particle1.velocity[2]*1e6:.1f}) μm/s")
    
    print(f"Particle 2 created / 粒子2已创建: radius={particle2.radius*1e9:.0f}nm, mass={particle2.mass*1e15:.2f}fg")
    print(f"Initial position 2 / 初始位置2: ({particle2.position[0]*1e6:.1f}, {particle2.position[1]*1e6:.1f}, {particle2.position[2]*1e6:.1f}) μm")
    print(f"Initial velocity 2 / 初始速度2: ({particle2.velocity[0]*1e6:.1f}, {particle2.velocity[1]*1e6:.1f}, {particle2.velocity[2]*1e6:.1f}) μm/s")
    
    # 计算初始粒子间距离 / Calculate initial inter-particle distance
    initial_distance = np.linalg.norm(particle1.position - particle2.position)
    print(f"Initial inter-particle distance / 初始粒子间距离: {initial_distance*1e6:.2f} μm")
    
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
        laser_power=0.25,  # 250mW (增加功率以处理两个粒子) / 250mW (increased power for two particles)
        w0=2e-6,  # 束腰半径2μm / Beam waist radius 2μm
        l=6  # 轨道角动量量子数 l=6 / Orbital angular momentum quantum number l=6
    )
    
    # 4. 设置LG61光场 / Set up LG61 optical field
    x_range = np.linspace(-10e-6, 10e-6, 80)  # 扩大范围以容纳两个轨道 / Expand range for two orbits
    y_range = np.linspace(-10e-6, 10e-6, 80)
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
        particles=particles,  # 传入粒子列表 / Pass particle list
        environment=environment,
        optical_trap=optical_trap
    )
    
    # 设置仿真参数 / Set simulation parameters
    sim_box.timestep = 1e-6  # 1μs时间步长 / 1μs time step
    sim_box.time = 0.0
    
    # 初始化阻尼系数 / Initialize damping coefficient
    sim_box.gamma = np.array([
        environment.get_drag_coefficient(particle1),
        environment.get_drag_coefficient(particle1),
        environment.get_drag_coefficient(particle1)
    ])
    
    print(f"Damping coefficient / 阻尼系数: γ={sim_box.gamma[0]:.2e} kg/s")
    
    # 6. 运行模拟 / Run simulation
    print("Starting simulation... / 开始模拟...")
    simulation_start_time = time.time()
    
    duration = 0.020  # 20ms模拟时间（增加以观察轨道运动）/ 20ms simulation time (increased to observe orbital motion)
    trajectory = sim_box.simulate(duration)
    
    simulation_end_time = time.time()
    simulation_elapsed = simulation_end_time - simulation_start_time
    
    print(f"Simulation completed / 模拟完成, total time / 总时间: {duration*1000}ms")
    print(f"Actual computation time / 实际计算时间: {simulation_elapsed:.3f}s")
    print(f"Time steps / 时间步数: {len(trajectory[0]['time'])}")
    
    # 7. 保存轨迹数据 / Save trajectory data
    import pandas as pd
    
    # 保存粒子1的轨迹 / Save particle 1 trajectory
    df1 = pd.DataFrame({
        'Time (s)': trajectory[0]['time'],
        'X (m)': trajectory[0]['position'][:, 0],
        'Y (m)': trajectory[0]['position'][:, 1],
        'Z (m)': trajectory[0]['position'][:, 2],
        'Vx (m/s)': trajectory[0]['velocity'][:, 0],
        'Vy (m/s)': trajectory[0]['velocity'][:, 1],
        'Vz (m/s)': trajectory[0]['velocity'][:, 2]
    })
    df1.to_csv('particle1_trajectory_lg61_two_particles.csv', index=False)
    
    # 保存粒子2的轨迹 / Save particle 2 trajectory
    df2 = pd.DataFrame({
        'Time (s)': trajectory[1]['time'],
        'X (m)': trajectory[1]['position'][:, 0],
        'Y (m)': trajectory[1]['position'][:, 1],
        'Z (m)': trajectory[1]['position'][:, 2],
        'Vx (m/s)': trajectory[1]['velocity'][:, 0],
        'Vy (m/s)': trajectory[1]['velocity'][:, 1],
        'Vz (m/s)': trajectory[1]['velocity'][:, 2]
    })
    df2.to_csv('particle2_trajectory_lg61_two_particles.csv', index=False)
    
    print("Trajectory data saved / 轨迹数据已保存:")
    print("- Particle 1: particle1_trajectory_lg61_two_particles.csv")
    print("- Particle 2: particle2_trajectory_lg61_two_particles.csv")
    
    # 8. 输出统计信息 / Output statistics
    print("\n=== Simulation Results Statistics / 模拟结果统计 ===")
    
    for i, traj in enumerate(trajectory):
        final_position = traj['position'][-1]
        max_displacement = np.max(np.linalg.norm(traj['position'], axis=1))
        mean_speed = np.mean(np.linalg.norm(traj['velocity'], axis=1))
        max_force = np.max(np.linalg.norm(traj['force'], axis=1))
        
        # 计算轨道半径变化 / Calculate orbital radius variation
        radii = np.sqrt(traj['position'][:, 0]**2 + traj['position'][:, 1]**2)
        initial_radius = radii[0]
        final_radius = radii[-1]
        mean_radius = np.mean(radii)
        radius_std = np.std(radii)
        
        print(f"\n--- Particle {i+1} / 粒子{i+1} ---")
        print(f"Final position / 最终位置: ({final_position[0]*1e6:.2f}, {final_position[1]*1e6:.2f}, {final_position[2]*1e6:.2f}) μm")
        print(f"Initial orbital radius / 初始轨道半径: {initial_radius*1e6:.2f} μm")
        print(f"Final orbital radius / 最终轨道半径: {final_radius*1e6:.2f} μm")
        print(f"Mean orbital radius / 平均轨道半径: {mean_radius*1e6:.2f} μm")
        print(f"Radius stability (std) / 轨道稳定性 (标准差): {radius_std*1e6:.3f} μm")
        print(f"Average velocity / 平均速度: {mean_speed*1e6:.2f} μm/s")
        print(f"Maximum force / 最大力: {max_force*1e12:.2f} pN")
        
        # 检查角运动 / Check angular motion
        if 'angular_velocity' in traj:
            max_angular_velocity = np.max(np.linalg.norm(traj['angular_velocity'], axis=1))
            print(f"Maximum angular velocity / 最大角速度: {max_angular_velocity:.2e} rad/s")
        
        if 'torque' in traj:
            max_torque = np.max(np.linalg.norm(traj['torque'], axis=1))
            print(f"Maximum torque / 最大扭矩: {max_torque*1e15:.2f} fN·m")
    
    # 分析粒子间相互作用 / Analyze inter-particle interactions
    final_distance = np.linalg.norm(trajectory[0]['position'][-1] - trajectory[1]['position'][-1])
    distance_change = final_distance - initial_distance
    
    print(f"\n--- Inter-particle Analysis / 粒子间相互作用分析 ---")
    print(f"Final inter-particle distance / 最终粒子间距离: {final_distance*1e6:.2f} μm")
    print(f"Distance change / 距离变化: {distance_change*1e6:.2f} μm")
    
    # 计算轨道运动特征 / Calculate orbital motion characteristics
    print(f"\n--- Orbital Motion Analysis / 轨道运动分析 ---")
    for i, traj in enumerate(trajectory):
        # 计算角位置 / Calculate angular position
        angles = np.arctan2(traj['position'][:, 1], traj['position'][:, 0])
        
        # 计算角速度（数值微分）/ Calculate angular velocity (numerical differentiation)
        dt = traj['time'][1] - traj['time'][0]
        angular_velocities = np.diff(angles) / dt
        
        # 处理角度跳跃 / Handle angle wrapping
        angular_velocities = np.where(angular_velocities > np.pi/dt, 
                                    angular_velocities - 2*np.pi/dt, angular_velocities)
        angular_velocities = np.where(angular_velocities < -np.pi/dt, 
                                    angular_velocities + 2*np.pi/dt, angular_velocities)
        
        mean_angular_velocity = np.mean(angular_velocities)
        rotation_frequency = mean_angular_velocity / (2 * np.pi)
        
        print(f"Particle {i+1} / 粒子{i+1}:")
        print(f"  Mean angular velocity / 平均角速度: {mean_angular_velocity:.3f} rad/s")
        print(f"  Rotation frequency / 旋转频率: {rotation_frequency:.3f} Hz")
        print(f"  Rotation period / 旋转周期: {1/abs(rotation_frequency):.3f} s" if rotation_frequency != 0 else "  No rotation detected / 未检测到旋转")
    
    return trajectory

def visualize_lg61_two_particles():
    """可视化LG61双粒子轨迹 / Visualize LG61 two particles trajectory"""
    import pandas as pd  # 在函数内部导入
    
    # 重新创建光阱用于可视化 / Recreate optical trap for visualization
    optical_trap = OpticalTrap(
        kappa=[1e-6, 1e-6, 1e-7],
        center=np.array([0.0, 0.0, 0.0]),
        wavelength=1064e-9,
        laser_power=0.25,
        w0=2e-6,
        l=6
    )
    
    # 重新设置光场用于可视化 / Reset optical field for visualization
    x_range = np.linspace(-10e-6, 10e-6, 60)  # 降低分辨率用于可视化 / Reduce resolution for visualization
    y_range = np.linspace(-10e-6, 10e-6, 60)
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
    
    # 创建可视化器并绘制双粒子轨迹 / Create visualizer and plot two particle trajectories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 绘制光场背景 / Plot optical field background
    X, Y = np.meshgrid(x_range, y_range)
    Z_field = np.zeros_like(X)
    field_intensity = lg61_field_function(X, Y, Z_field)
    
    # 左图：粒子1轨迹 / Left plot: Particle 1 trajectory
    im1 = ax1.contourf(X*1e6, Y*1e6, field_intensity, levels=20, alpha=0.6, cmap='viridis')
    
    # 加载并绘制粒子1轨迹 / Load and plot particle 1 trajectory
    try:
        df1 = pd.read_csv('particle1_trajectory_lg61_two_particles.csv')
        ax1.plot(df1['X (m)']*1e6, df1['Y (m)']*1e6, 'r-', linewidth=2, label='Particle 1 Trajectory')
        ax1.plot(df1['X (m)'].iloc[0]*1e6, df1['Y (m)'].iloc[0]*1e6, 'ro', markersize=8, label='Start')
        ax1.plot(df1['X (m)'].iloc[-1]*1e6, df1['Y (m)'].iloc[-1]*1e6, 'rs', markersize=8, label='End')
    except FileNotFoundError:
        print("Particle 1 trajectory file not found / 粒子1轨迹文件未找到")
    
    ax1.set_xlabel('X (μm)')
    ax1.set_ylabel('Y (μm)')
    ax1.set_title('LG61 Beam - Particle 1 (Inner Orbit) / LG61光束 - 粒子1 (内轨道)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 右图：粒子2轨迹 / Right plot: Particle 2 trajectory
    im2 = ax2.contourf(X*1e6, Y*1e6, field_intensity, levels=20, alpha=0.6, cmap='viridis')
    
    # 加载并绘制粒子2轨迹 / Load and plot particle 2 trajectory
    try:
        df2 = pd.read_csv('particle2_trajectory_lg61_two_particles.csv')
        ax2.plot(df2['X (m)']*1e6, df2['Y (m)']*1e6, 'b-', linewidth=2, label='Particle 2 Trajectory')
        ax2.plot(df2['X (m)'].iloc[0]*1e6, df2['Y (m)'].iloc[0]*1e6, 'bo', markersize=8, label='Start')
        ax2.plot(df2['X (m)'].iloc[-1]*1e6, df2['Y (m)'].iloc[-1]*1e6, 'bs', markersize=8, label='End')
    except FileNotFoundError:
        print("Particle 2 trajectory file not found / 粒子2轨迹文件未找到")
    
    ax2.set_xlabel('X (μm)')
    ax2.set_ylabel('Y (μm)')
    ax2.set_title('LG61 Beam - Particle 2 (Outer Orbit) / LG61光束 - 粒子2 (外轨道)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('lg61_two_particles_trajectories.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved / 可视化已保存: lg61_two_particles_trajectories.png")

if __name__ == "__main__":
    # 运行LG61双粒子测试 / Run LG61 two particles test
    trajectory = test_lg61_two_particles()
    
    # 可视化结果 / Visualize results
    visualize_lg61_two_particles()
    
    print("\n=== LG61 Two Particles Test Completed / LG61双粒子测试完成 ===")
    print("Results saved / 结果已保存:")
    print("- Particle 1 trajectory / 粒子1轨迹: particle1_trajectory_lg61_two_particles.csv")
    print("- Particle 2 trajectory / 粒子2轨迹: particle2_trajectory_lg61_two_particles.csv")
    print("- Visualization / 可视化: lg61_two_particles_trajectories.png")