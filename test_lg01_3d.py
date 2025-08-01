import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加simulation模块到路径 / Add simulation module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

from particle import ParticleFactory
from environment import Environment
from trap import OpticalTrap
from box import SimulationBox
from visualizer import TrajectoryVisualizer

def create_lg01_field(r, phi, z, w0=2e-6, wavelength=1064e-9, l=1, p=0):
    """创建LG01光束的强度分布"""
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
    
    # 高斯包络 / Gaussian envelope
    gaussian_envelope = (w0 / w_z) * np.exp(-r**2 / w_z**2)
    
    # 完整的LG01场强度 / Complete LG01 field intensity
    intensity = (gaussian_envelope * radial_part)**2
    
    # 返回强度，而不是复数振幅
    return intensity

def test_lg01_3d_visualization():
    """测试LG01光束中粒子运动的3D可视化 / Test 3D visualization of particle motion in LG01 beam"""
    
    print("开始LG01 3D可视化测试... / Starting LG01 3D visualization test...")
    
    # 1. 创建粒子 / Create particle
    particle = ParticleFactory.create_polystyrene_sphere(
        radius=500e-9,  # 500nm聚苯乙烯球 / 500nm polystyrene sphere
        position=np.array([1.0e-6, 0.5e-6, 0.2e-6])  # 初始位置在3D空间中 / Initial position in 3D space
    )
    print(f"创建粒子: 半径={particle.radius*1e9:.1f}nm, 质量={particle.mass*1e15:.2f}fg / Created particle: radius={particle.radius*1e9:.1f}nm, mass={particle.mass*1e15:.2f}fg")
    
    # 2. 创建环境（水环境）/ Create environment (water medium)
    environment = Environment(
        medium='liquid',
        T=298.0,  # 室温 / Room temperature
        eta=0.001  # 水的粘度 / Water viscosity
    )
    print(f"环境设置: {environment.medium}, T={environment.T}K / Environment setup: {environment.medium}, T={environment.T}K")
    
    # 3. 创建LG01光阱 / Create LG01 optical trap
    optical_trap = OpticalTrap(
        kappa=[1e-5, 1e-5, 1e-5],  # 阱刚度 [N/m] / Trap stiffness [N/m]
        center=np.array([0.0, 0.0, 0.0]),
        wavelength=1064e-9,  # 1064nm激光 / 1064nm laser
        laser_power=10,  # 10mW
        w0=2e-6,  # 2μm束腰 / 2μm beam waist
        l=1  # LG01的轨道角动量量子数 / Orbital angular momentum quantum number for LG01
    )
    
    # 4. 设置LG01光场 / Setup LG01 optical field
    # 创建三维网格 / Create 3D grid
    x_range = np.linspace(-5e-6, 5e-6, 50)  # ±5μm
    y_range = np.linspace(-5e-6, 5e-6, 50)
    z_range = np.linspace(-3e-6, 3e-6, 30)  # 扩展Z范围用于3D可视化 / Extended Z range for 3D visualization
    
    def lg01_field_function(x, y, z):
        """LG01光场函数 / LG01 optical field function"""
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return create_lg01_field(r, phi, z, w0=optical_trap.w0, 
                               wavelength=optical_trap.wavelength, l=1, p=0)
    
    def lg01_phase_function(x, y, z):
        """LG01光束的相位函数 / LG01 beam phase function"""
        phi = np.arctan2(y, x)
        return optical_trap.l * phi  # 返回轨道角动量相位
    
    # 设置光场 / Set optical field
    optical_trap.set_field(x_range, y_range, z_range, 
                          lg01_field_function, lg01_phase_function)
    print("LG01光场设置完成 / LG01 optical field setup completed")
    
    # 5. 创建模拟盒子 / Create simulation box
    sim_box = SimulationBox(
        particles=particle,
        environment=environment,
        optical_trap=optical_trap
    )
    
    # 设置仿真参数 / Set simulation parameters
    sim_box.timestep = 1e-3  # 1ms时间步长 / 1ms time step
    sim_box.time = 0.0
    
    # 初始化阻尼系数 / Initialize damping coefficient
    sim_box.gamma = np.array([
        environment.get_drag_coefficient(particle),
        environment.get_drag_coefficient(particle),
        environment.get_drag_coefficient(particle)
    ])
    
    print(f"阻尼系数: γ={sim_box.gamma[0]:.2e} kg/s / Damping coefficient: γ={sim_box.gamma[0]:.2e} kg/s")
    
    # 6. 运行模拟 / Run simulation
    print("开始模拟... / Starting simulation...")
    duration = 3  # 300ms模拟时间，更长的时间用于3D轨迹 / 300ms simulation time, longer for 3D trajectory
    trajectory = sim_box.simulate(duration)
    
    print(f"模拟完成，总时间: {duration}ms / Simulation completed, total time: {duration}ms")
    print(f"时间步数: {len(trajectory[0]['time'])} / Number of time steps: {len(trajectory[0]['time'])}")
    
    # 7. 保存结果 / Save results
    output_file = "particle_trajectory_lg01_3d_test.csv"
    sim_box.save_trajectory_to_csv(output_file)
    print(f"轨迹数据已保存到: {output_file} / Trajectory data saved to: {output_file}")
    
    # 8. 输出统计信息 / Output statistics
    final_position = trajectory[0]['position'][-1]
    max_displacement = np.max(np.linalg.norm(trajectory[0]['position'], axis=1))
    mean_speed = np.mean(np.linalg.norm(trajectory[0]['velocity'], axis=1))
    max_force = np.max(np.linalg.norm(trajectory[0]['force'], axis=1))
    max_angular_velocity = np.max(np.linalg.norm(trajectory[0]['angular_velocity'], axis=1))
    
    print("\n=== 3D模拟结果统计 / 3D Simulation Results Statistics ===")
    print(f"最终位置 / Final position: ({final_position[0]*1e6:.2f}, {final_position[1]*1e6:.2f}, {final_position[2]*1e6:.2f}) μm")
    print(f"最大位移 / Maximum displacement: {max_displacement*1e6:.2f} μm")
    print(f"平均速度 / Average velocity: {mean_speed*1e6:.2f} μm/s")
    print(f"最大受力 / Maximum force: {max_force*1e12:.2f} pN")
    print(f"最大角速度 / Maximum angular velocity: {max_angular_velocity:.2e} rad/s")
    
    # 9. 创建3D可视化 / Create 3D visualization
    print("\n开始3D可视化... / Starting 3D visualization...")
    
    # 创建可视化器并加载数据 / Create visualizer and load data
    visualizer = TrajectoryVisualizer(output_file)
    
    # 绘制3D轨迹图 / Plot 3D trajectory
    print("绘制3D轨迹图... / Plotting 3D trajectory...")
    visualizer.plot_3d_trajectory(figsize=(14, 10))
    
    # 也可以绘制2D投影进行对比 / Also plot 2D projections for comparison
    print("绘制2D投影图... / Plotting 2D projections...")
    
    # XY平面投影 / XY plane projection
    visualizer.plot_2d_trajectory('xy', figsize=(10, 8))
    
    # XZ平面投影 / XZ plane projection
    visualizer.plot_2d_trajectory('xz', figsize=(10, 8))
    
    # YZ平面投影 / YZ plane projection
    visualizer.plot_2d_trajectory('yz', figsize=(10, 8))
    
    return trajectory, sim_box, visualizer

if __name__ == "__main__":
    # 运行3D可视化测试 / Run 3D visualization test
    trajectory, sim_box, visualizer = test_lg01_3d_visualization()
    
    print("\n=== 3D可视化测试完成！ / 3D Visualization Test Completed! ===")
    print("已生成以下图形: / Generated the following plots:")
    print("1. 3D轨迹图 / 3D trajectory plot")
    print("2. XY平面投影 / XY plane projection")
    print("3. XZ平面投影 / XZ plane projection")
    print("4. YZ平面投影 / YZ plane projection")
    print("\n你可以使用以下命令重新绘制3D轨迹: / You can redraw the 3D trajectory using:")
    print("visualizer.plot_3d_trajectory(figsize=(14, 10))")