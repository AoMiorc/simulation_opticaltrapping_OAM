import numpy as np
import sys
import os
import time

# 添加simulation模块到路径 / Add simulation module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

from particle import ParticleFactory
from environment import Environment
from trap import OpticalTrap
from box import SimulationBox

def create_lg01_field(r, phi, z, w0=2e-6, wavelength=1064e-9, l=1, p=0):
    """
    创建LG01光束的强度分布 / Create LG01 beam intensity distribution
    
    参数 / Parameters:
    r, phi, z: 柱坐标系坐标 / Cylindrical coordinates
    w0: 束腰半径 / Beam waist radius
    wavelength: 波长 / Wavelength
    l: 轨道角动量量子数 (对于LG01，l=1) / Orbital angular momentum quantum number (for LG01, l=1)
    p: 径向量子数 (对于LG01，p=0) / Radial quantum number (for LG01, p=0)
    """
    # 避免除零错误 / Avoid division by zero
    r = np.maximum(r, 1e-12)  # 设置最小半径值 / Set minimum radius value
    
    # 瑞利长度 / Rayleigh length
    z_R = np.pi * w0**2 / wavelength
    
    # 束腰随z变化 / Beam waist variation with z
    w_z = w0 * np.sqrt(1 + (z / z_R)**2)
    
    # 归一化径向坐标 / Normalized radial coordinate
    rho = np.sqrt(2) * r / w_z
    
    # LG01模式的径向部分 (l=1, p=0) / Radial part of LG01 mode (l=1, p=0)
    # 对于p=0，广义拉盖尔多项式L_0^1(ρ²) = 1 / For p=0, generalized Laguerre polynomial L_0^1(ρ²) = 1
    radial_part = rho * np.exp(-rho**2 / 2)
    
    # 角向部分 (l=1) / Angular part (l=1)
    angular_part = np.exp(1j * l * phi)
    
    # 高斯包络 / Gaussian envelope
    gaussian_envelope = (w0 / w_z) * np.exp(-r**2 / w_z**2)
    
    # 完整的LG01场振幅 / Complete LG01 field amplitude
    amplitude = gaussian_envelope * radial_part * angular_part
    
    # 返回强度 |E|²，确保结果为实数且非负 / Return intensity |E|², ensure real and non-negative result
    intensity = np.abs(amplitude)**2
    return np.real(intensity)

def test_lg01_two_particles():
    """测试两个粒子在LG01光束中的相互作用和运动 / Test interaction and motion of two particles in LG01 beam"""
    
    print("=== LG01 Two Particles Optical Trap Test / LG01双粒子光阱测试 ===")
    
    # 记录测试开始时间 / Record test start time
    test_start_time = time.time()
    
    # 1. 创建两个粒子 / Create two particles
    particle1 = ParticleFactory.create_polystyrene_sphere(
        radius=500e-9,  # 500nm聚苯乙烯球 / 500nm polystyrene sphere
        position=np.array([1.0e-6, 0.5e-6, 0.0])  # 粒子1初始位置 / Initial position of particle 1
    )
    particle1.velocity = np.array([0.0, 0.0, 0.0])  # 初始静止 / Initially at rest
    
    particle2 = ParticleFactory.create_polystyrene_sphere(
        radius=500e-9,  # 500nm聚苯乙烯球 / 500nm polystyrene sphere
        position=np.array([-1.0e-6, -0.5e-6, 0.0])  # 粒子2初始位置 / Initial position of particle 2
    )
    particle2.velocity = np.array([0.0, 0.0, 0.0])  # 初始静止 / Initially at rest
    
    particles = [particle1, particle2]
    
    print(f"Particle 1 created / 粒子1已创建: radius={particle1.radius*1e9:.1f}nm, mass={particle1.mass*1e15:.2f}fg")
    print(f"Initial position 1 / 初始位置1: ({particle1.position[0]*1e6:.1f}, {particle1.position[1]*1e6:.1f}, {particle1.position[2]*1e6:.1f}) μm")
    print(f"Particle 2 created / 粒子2已创建: radius={particle2.radius*1e9:.1f}nm, mass={particle2.mass*1e15:.2f}fg")
    print(f"Initial position 2 / 初始位置2: ({particle2.position[0]*1e6:.1f}, {particle2.position[1]*1e6:.1f}, {particle2.position[2]*1e6:.1f}) μm")
    
    # 计算初始粒子间距离 / Calculate initial inter-particle distance
    initial_distance = np.linalg.norm(particle1.position - particle2.position)
    print(f"Initial inter-particle distance / 初始粒子间距离: {initial_distance*1e6:.2f} μm")
    
    # 2. 创建环境（水环境）/ Create environment (water medium)
    environment = Environment(
        medium='liquid',
        T=298.0,  # 室温 / Room temperature
        eta=0.001  # 水的粘度 / Water viscosity
    )
    print(f"Environment setup / 环境设置: {environment.medium}, T={environment.T}K")
    
    # 3. 创建LG01光阱 / Create LG01 optical trap
    optical_trap = OpticalTrap(
        kappa=[1e-6, 1e-6, 1e-7],  # 阱刚度 [N/m] / Trap stiffness [N/m]
        center=np.array([0.0, 0.0, 0.0]),
        wavelength=1064e-9,  # 1064nm激光 / 1064nm laser
        laser_power=0.15,  # 150mW (稍微增加功率以处理两个粒子) / 150mW (slightly increased power for two particles)
        w0=2e-6,  # 2μm束腰 / 2μm beam waist
        l=1  # LG01的轨道角动量量子数 / Orbital angular momentum quantum number for LG01
    )
    
    # 4. 设置LG01光场 / Set up LG01 optical field
    # 创建三维网格 / Create 3D grid
    x_range = np.linspace(-6e-6, 6e-6, 60)  # 稍微扩大范围以容纳两个粒子 / Slightly expanded range to accommodate two particles
    y_range = np.linspace(-6e-6, 6e-6, 60)
    z_range = np.linspace(-2e-6, 2e-6, 20)  # ±2μm
    
    def lg01_field_function(x, y, z):
        """LG01光场函数 / LG01 optical field function"""
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return create_lg01_field(r, phi, z, w0=optical_trap.w0, 
                               wavelength=optical_trap.wavelength, l=1, p=0)
    
    def lg01_phase_function(x, y, z):
        """LG01相位函数 / LG01 phase function"""
        phi = np.arctan2(y, x)
        return optical_trap.l * phi  # l=1的相位 / Phase for l=1
    
    # 设置光场 / Set optical field
    optical_trap.set_field(x_range, y_range, z_range, 
                          lg01_field_function, lg01_phase_function)
    print("LG01 field setup completed / LG01光场设置完成")
    
    # 5. 创建模拟盒子（传入粒子列表）/ Create simulation box (pass particle list)
    sim_box = SimulationBox(
        particles=particles,  # 传入粒子列表 / Pass particle list
        environment=environment,
        optical_trap=optical_trap
    )
    
    # 设置仿真参数 / Set simulation parameters
    sim_box.timestep = 1e-6  # 1μs时间步长 / 1μs time step
    sim_box.time = 0.0
    
    # 初始化阻尼系数（对每个粒子）/ Initialize damping coefficient (for each particle)
    sim_box.gamma = np.array([
        environment.get_drag_coefficient(particle1),
        environment.get_drag_coefficient(particle1),
        environment.get_drag_coefficient(particle1)
    ])
    
    print(f"Damping coefficient / 阻尼系数: γ={sim_box.gamma[0]:.2e} kg/s")
    
    # 6. 运行模拟 / Run simulation
    print("Starting simulation... / 开始模拟...")
    simulation_start_time = time.time()
    
    duration = 0.015  # 15ms模拟时间（稍微增加以观察粒子相互作用）/ 15ms simulation time (slightly increased to observe particle interactions)
    trajectory = sim_box.simulate(duration)
    
    simulation_end_time = time.time()
    simulation_elapsed = simulation_end_time - simulation_start_time
    
    print(f"Simulation completed / 模拟完成, total time / 总时间: {duration*1000}ms")
    print(f"Actual computation time / 实际计算时间: {simulation_elapsed:.3f}s")
    print(f"Time steps / 时间步数: {len(trajectory[0]['time'])}")
    
    # 7. 保存结果 / Save results
    output_file = "particle_trajectory_lg01_two_particles.csv"
    sim_box.save_trajectory_to_csv(output_file)
    print(f"Trajectory data saved to / 轨迹数据已保存至: {output_file}")
    
    # 8. 输出统计信息 / Output statistics
    print("\n=== Simulation Results Statistics / 模拟结果统计 ===")
    
    for i, traj in enumerate(trajectory):
        final_position = traj['position'][-1]
        max_displacement = np.max(np.linalg.norm(traj['position'], axis=1))
        mean_speed = np.mean(np.linalg.norm(traj['velocity'], axis=1))
        max_force = np.max(np.linalg.norm(traj['force'], axis=1))
        
        print(f"\n--- Particle {i+1} / 粒子{i+1} ---")
        print(f"Final position / 最终位置: ({final_position[0]*1e6:.2f}, {final_position[1]*1e6:.2f}, {final_position[2]*1e6:.2f}) μm")
        print(f"Maximum displacement / 最大位移: {max_displacement*1e6:.2f} μm")
        print(f"Average velocity / 平均速度: {mean_speed*1e6:.2f} μm/s")
        print(f"Maximum force / 最大力: {max_force*1e12:.2f} pN")
        
        # 检查角运动 / Check angular motion
        max_angular_velocity = np.max(np.linalg.norm(traj['angular_velocity'], axis=1))
        max_torque = np.max(np.linalg.norm(traj['torque'], axis=1))
        print(f"Maximum angular velocity / 最大角速度: {max_angular_velocity:.2e} rad/s")
        print(f"Maximum torque / 最大扭矩: {max_torque*1e15:.2f} fN·m")
    
    # 分析粒子间相互作用 / Analyze inter-particle interactions
    final_distance = np.linalg.norm(trajectory[0]['position'][-1] - trajectory[1]['position'][-1])
    distance_change = final_distance - initial_distance
    
    print(f"\n--- Inter-particle Analysis / 粒子间相互作用分析 ---")
    print(f"Final inter-particle distance / 最终粒子间距离: {final_distance*1e6:.2f} μm")
    print(f"Distance change / 距离变化: {distance_change*1e6:.2f} μm")
    
    # 计算质心运动 / Calculate center of mass motion
    com_trajectory = (trajectory[0]['position'] + trajectory[1]['position']) / 2
    com_displacement = np.linalg.norm(com_trajectory[-1] - com_trajectory[0])
    print(f"Center of mass displacement / 质心位移: {com_displacement*1e6:.2f} μm")
    
    # 计算总测试时间 / Calculate total test time
    test_end_time = time.time()
    total_elapsed = test_end_time - test_start_time
    print(f"\n=== Timing Statistics / 时间统计 ===")
    print(f"Total test time / 总测试时间: {total_elapsed:.3f}s")
    print(f"Simulation time / 模拟时间: {simulation_elapsed:.3f}s ({simulation_elapsed/total_elapsed*100:.1f}% of total / 占总时间)")
    print(f"Setup and analysis time / 设置和分析时间: {total_elapsed-simulation_elapsed:.3f}s")
    
    return trajectory, sim_box

if __name__ == "__main__":
    # 运行测试并捕获返回值 / Run test and capture return values
    trajectory, sim_box = test_lg01_two_particles()
    
    # 创建可视化器并加载数据 / Create visualizer and load data
    from simulation.visualizer import TrajectoryVisualizer
    visualizer = TrajectoryVisualizer("particle_trajectory_lg01_two_particles.csv")
    
    # 重新创建光阱对象用于可视化 / Recreate optical trap object for visualization
    optical_trap = OpticalTrap(
        kappa=[1e-6, 1e-6, 1e-7],
        center=np.array([0.0, 0.0, 0.0]),
        wavelength=1064e-9,
        laser_power=0.15,
        w0=2e-6,
        l=1
    )
    
    # 重新设置光场 / Reset optical field
    x_range = np.linspace(-6e-6, 6e-6, 60)
    y_range = np.linspace(-6e-6, 6e-6, 60)
    z_range = np.linspace(-2e-6, 2e-6, 20)
    
    def lg01_field_function(x, y, z):
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return create_lg01_field(r, phi, z, w0=optical_trap.w0, 
                               wavelength=optical_trap.wavelength, l=1, p=0)
    
    optical_trap.set_field(x_range, y_range, z_range, lg01_field_function)
    
    # 绘制带有光场背景的轨迹图 / Plot trajectory with optical field background
    visualizer.plot_2d_trajectory_with_field('xy', optical_trap=optical_trap, field_alpha=0.6)
    
    print("\nTwo-particle LG01 test completed! / 双粒子LG01测试完成！")
    print("This test demonstrates particle-particle interactions in LG01 beam. / 此测试演示了LG01光束中的粒子间相互作用。")
    print("You can use the TrajectoryVisualizer class in visualizer.py to visualize the results. / 您可以使用visualizer.py中的TrajectoryVisualizer类来可视化结果。")