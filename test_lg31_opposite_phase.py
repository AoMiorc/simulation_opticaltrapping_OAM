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

def create_lg31_opposite_phase_field(r, phi, z, w0=2e-6, wavelength=1064e-9, l=3, p=1):
    """
    创建LG31光束强度分布，内环和外环相位相反 (l=3, p=1) 
    Create LG31 beam intensity distribution with opposite phase for inner and outer rings (l=3, p=1)
    
    参数 / Parameters:
    r, phi, z: 柱坐标系坐标 / Cylindrical coordinates
    w0: 束腰半径 (beam waist) / Beam waist radius
    wavelength: 波长 / Wavelength
    l: 轨道角动量量子数 (azimuthal index) = 3 / Orbital angular momentum quantum number (azimuthal index) = 3
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
    
    # 径向项 r^|l| = r^3 / Radial term r^|l| = r^3
    radial_power = (r / w_z) ** abs(l)
    
    # 广义拉盖尔多项式 L_1^3(ρ²) = 1 + 3 - ρ² = 4 - ρ² / Generalized Laguerre polynomial L_1^3(ρ²) = 1 + 3 - ρ² = 4 - ρ²
    laguerre_term = 4 - rho_squared
    
    # 高斯包络 / Gaussian envelope
    gaussian_envelope = np.exp(-rho_squared / 2)
    
    # 关键修改：内环和外环使用相反的l值 / Key modification: opposite l values for inner and outer rings
    boundary_radius = 3.5e-6
    
    # 内环：l = +3 (逆时针旋转)，外环：l = -3 (顺时针旋转) / Inner ring: l = +3 (counterclockwise), outer ring: l = -3 (clockwise)
    effective_l = np.where(r < boundary_radius, l, -l)
    
    # 角向相位（产生相反的涡旋）/ Angular phase (creates opposite vortices)
    angular_phase = np.exp(1j * effective_l * phi)
    
    # 完整的LG场振幅 / Complete LG field amplitude
    amplitude = radial_power * laguerre_term * gaussian_envelope * angular_phase
    
    # 强度 I = |E|² / Intensity I = |E|²
    intensity = np.abs(amplitude)**2
    
    # 归一化强度 / Normalize intensity
    if np.max(intensity) > 0:
        intensity = intensity / np.max(intensity)
    
    return intensity

def test_lg31_opposite_phase_two_particles():
    """测试相位相反的LG31光束中的双粒子动力学 / Test two particle dynamics in opposite phase LG31 beam"""
    print("=== LG31 Opposite Phase Beam Two Particles Test / LG31相位相反光束双粒子测试 ===")
    
    # 记录测试开始时间 / Record test start time
    test_start_time = time.time()
    
    # 1. 创建两个粒子，分别放在不同的轨道上 / Create two particles on different orbits
    # 粒子1：放在内轨道（第一个亮环）/ Particle 1: on inner orbit (first bright ring)
    particle1 = ParticleFactory.create_particle(
        material='polystyrene',
        radius=400e-9,  # 400nm半径 / 400nm radius
        position=np.array([3.0e-6, 0.0e-6, 0.0])  # 内轨道位置 / Inner orbit position
    )
    # 给粒子1初始切向速度，使其在轨道上运动 / Give particle 1 initial tangential velocity
    particle1.velocity = np.array([0.0, 15e-6, 0.0])  # 切向速度 15μm/s / Tangential velocity 15μm/s
    
    # 粒子2：放在外轨道（第二个亮环）/ Particle 2: on outer orbit (second bright ring)
    particle2 = ParticleFactory.create_particle(
        material='silica',
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
    
    # 3. 创建LG31光阱 / Create LG31 optical trap
    optical_trap = OpticalTrap(
        kappa=[1e-6, 1e-6, 1e-7],  # 阱刚度 [N/m] / Trap stiffness [N/m]
        center=np.array([0.0, 0.0, 0.0]),
        wavelength=1064e-9,  # 1064nm激光 / 1064nm laser
        laser_power=0.25,  # 250mW (增加功率以处理两个粒子) / 250mW (increased power for two particles)
        w0=2e-6,  # 束腰半径2μm / Beam waist radius 2μm
        l=3  # 轨道角动量量子数 l=3 / Orbital angular momentum quantum number l=3
    )
    
    # 4. 设置相位相反的LG31光场 / Set up opposite phase LG31 optical field
    x_range = np.linspace(-10e-6, 10e-6, 80)  # 扩大范围以容纳两个轨道 / Expand range for two orbits
    y_range = np.linspace(-10e-6, 10e-6, 80)
    z_range = np.linspace(-2e-6, 2e-6, 20)  # ±2μm
    
    def lg31_opposite_phase_field_function(x, y, z):
        """相位相反的LG31光场函数 / Opposite phase LG31 optical field function"""
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return create_lg31_opposite_phase_field(r, phi, z, w0=optical_trap.w0, 
                                wavelength=optical_trap.wavelength, l=3, p=1)
    
    def lg31_opposite_phase_function(x, y, z):
        """相位相反的LG31相位函数 / Opposite phase LG31 phase function"""
        phi = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2)
        boundary_radius = 3.5e-6
        
        # 内环和外环使用相反的l值 / Opposite l values for inner and outer rings
        effective_l = np.where(r < boundary_radius, optical_trap.l, -optical_trap.l)
        return effective_l * phi  # l=3的相位 / Phase for l=3
    
    # 设置光场 / Set optical field
    optical_trap.set_field(x_range, y_range, z_range, 
                          lg31_opposite_phase_field_function, lg31_opposite_phase_function)
    print("LG31 opposite phase field setup completed / LG31相位相反光场设置完成")
    
    # 5. 创建模拟盒子 / Create simulation box
    sim_box = SimulationBox(
        particles=particles,  # 传入粒子列表 / Pass particle list
        environment=environment,
        optical_trap=optical_trap
    )
    
    # 设置仿真参数 / Set simulation parameters
    sim_box.timestep = 1e-3  # 1μs时间步长 / 1μs time step
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
    
    duration = 20  # 20ms模拟时间（增加以观察轨道运动）/ 20ms simulation time (increased to observe orbital motion)
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
        'Particle_ID': 1,
        'Time (s)': trajectory[0]['time'],
        'X (m)': trajectory[0]['position'][:, 0],
        'Y (m)': trajectory[0]['position'][:, 1],
        'Z (m)': trajectory[0]['position'][:, 2],
        'Vx (m/s)': trajectory[0]['velocity'][:, 0],
        'Vy (m/s)': trajectory[0]['velocity'][:, 1],
        'Vz (m/s)': trajectory[0]['velocity'][:, 2]
    })
    df1.to_csv('particle1_trajectory_lg31_opposite_phase.csv', index=False)
    
    # 保存粒子2的轨迹 / Save particle 2 trajectory
    df2 = pd.DataFrame({
        'Particle_ID': 2,
        'Time (s)': trajectory[1]['time'],
        'X (m)': trajectory[1]['position'][:, 0],
        'Y (m)': trajectory[1]['position'][:, 1],
        'Z (m)': trajectory[1]['position'][:, 2],
        'Vx (m/s)': trajectory[1]['velocity'][:, 0],
        'Vy (m/s)': trajectory[1]['velocity'][:, 1],
        'Vz (m/s)': trajectory[1]['velocity'][:, 2]
    })
    df2.to_csv('particle2_trajectory_lg31_opposite_phase.csv', index=False)
    
    # 合并轨迹数据到单个CSV / Merge trajectory data into a single CSV
    combined_df = pd.concat([df1, df2])
    combined_csv = 'combined_trajectory_lg31_opposite_phase.csv'
    combined_df.to_csv(combined_csv, index=False)
    
    print("Trajectory data saved / 轨迹数据已保存:")
    print("- Particle 1: particle1_trajectory_lg31_opposite_phase.csv")
    print("- Particle 2: particle2_trajectory_lg31_opposite_phase.csv")
    print("- Combined: combined_trajectory_lg31_opposite_phase.csv")
    
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
            print(f"Maximum torque / 最大扭矩: {max_torque:.2e} N·m")
    
    # 9. 可视化轨迹 / Visualize trajectory
    visualizer = TrajectoryVisualizer(combined_csv)
    visualizer.plot_3d_trajectory()  # 使用3D轨迹绘图方法 / Use 3D trajectory plot method
    plt.savefig('lg31_opposite_phase_two_particles_trajectories.png')
    plt.show()
    
    # 记录测试结束时间 / Record test end time
    test_end_time = time.time()
    total_elapsed = test_end_time - test_start_time
    print(f"\nTotal test time / 测试总时间: {total_elapsed:.3f}s")

if __name__ == "__main__":
    test_lg31_opposite_phase_two_particles()