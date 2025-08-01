import os
import sys
import numpy as np

# 添加simulation模块到路径 / Add simulation module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

from particle import ParticleFactory
from environment import Environment
from trap import OpticalTrap
from box import SimulationBox

def load_lp71_intensity_from_csv(filename):
    """
    从CSV文件加载LP71光场强度数据 / Load LP71 optical field intensity data from CSV file
    
    参数 / Parameters:
    filename: CSV文件路径 / CSV file path
    
    返回 / Returns:
    intensity_data: 强度数据数组 / Intensity data array
    """
    try:
        # 尝试加载CSV数据 / Try to load CSV data
        intensity_data = np.loadtxt(filename, delimiter=',')
        print(f"成功加载LP71光场数据，数据形状: {intensity_data.shape} / Successfully loaded LP71 field data, shape: {intensity_data.shape}")
        return intensity_data
    except Exception as e:
        print(f"加载CSV文件失败: {e} / Failed to load CSV file: {e}")
        print("使用理论LP71光场模型 / Using theoretical LP71 field model")
        return None

def create_lp71_field(r, phi, z, w0=2e-6, wavelength=1064e-9, l=7, p=1):
    """
    创建LP71光束的强度分布 / Create intensity distribution of LP71 beam
    
    参数 / Parameters:
    r, phi, z: 柱坐标系坐标 / Cylindrical coordinates
    w0: 束腰半径 / Beam waist radius
    wavelength: 波长 / Wavelength
    l: 轨道角动量量子数 (对于LP71，l=7) / Orbital angular momentum quantum number (for LP71, l=7)
    p: 径向量子数 (对于LP71，p=1) / Radial quantum number (for LP71, p=1)
    """
    # 避免除零错误 / Avoid division by zero
    r = np.maximum(r, 1e-12)  # 设置最小半径值 / Set minimum radius value
    
    # 瑞利长度 / Rayleigh length
    z_R = np.pi * w0**2 / wavelength
    
    # 束腰随z变化 / Beam waist variation with z
    w_z = w0 * np.sqrt(1 + (z / z_R)**2)
    
    # 归一化径向坐标 / Normalized radial coordinate
    rho = np.sqrt(2) * r / w_z
    
    # LP71模式的径向部分 (l=7, p=1) / Radial part of LP71 mode (l=7, p=1)
    # 广义拉盖尔多项式L_1^7(ρ²) = (8-ρ²) / Generalized Laguerre polynomial L_1^7(ρ²) = (8-ρ²)
    laguerre_part = 8 - rho**2
    radial_part = rho**7 * laguerre_part * np.exp(-rho**2 / 2)
    
    # 角向部分 (l=7) / Angular part (l=7)
    angular_part = np.exp(1j * l * phi)
    
    # 高斯包络 / Gaussian envelope
    gaussian_envelope = (w0 / w_z) * np.exp(-r**2 / w_z**2)
    
    # 完整的LP71场振幅 / Complete LP71 field amplitude
    amplitude = gaussian_envelope * radial_part * angular_part
    
    # 返回强度 |E|²，确保结果为实数且非负 / Return intensity |E|², ensure real and non-negative result
    intensity = np.abs(amplitude)**2
    return np.real(intensity)

def test_lp71_single_particle():
    """测试单粒子在LP71光束中的运动 / Test single particle motion in LP71 beam"""
    
    print("开始LP71单粒子测试... / Starting LP71 single particle test...")
    
    # 尝试加载CSV光场数据 / Try to load CSV field data
    csv_filename = "final_intensity_LP71.csv"
    intensity_data = load_lp71_intensity_from_csv(csv_filename)
    
    # 1. 创建粒子 / Create particle
    particle = ParticleFactory.create_polystyrene_sphere(
        radius=500e-9,  # 500nm聚苯乙烯球 / 500nm polystyrene sphere
        position=np.array([9e-6, 0.0, 0.0])  # 初始位置稍微偏离中心 / Initial position slightly off center
    )
    print(f"创建粒子: 半径={particle.radius*1e9:.1f}nm, 质量={particle.mass*1e15:.2f}fg / Created particle: radius={particle.radius*1e9:.1f}nm, mass={particle.mass*1e15:.2f}fg")
    
    # 2. 创建环境（水环境）/ Create environment (water medium)
    environment = Environment(
        medium='liquid',
        T=298.0,  # 室温 / Room temperature
        eta=0.001  # 水的粘度 / Water viscosity
    )
    print(f"环境设置: {environment.medium}, T={environment.T}K / Environment setup: {environment.medium}, T={environment.T}K")
    
    # 3. 创建LP71光阱 / Create LP71 optical trap
    optical_trap = OpticalTrap(
        kappa=[2e-7, 2e-7, 1e-7],  # 阱刚度 [N/m] / Trap stiffness [N/m]
        center=np.array([0.0, 0.0, 0.0]),
        wavelength=1064e-9,  # 1064nm激光 / 1064nm laser
        laser_power=1.5,  # 150mW，LP71需要更高功率 / 150mW, LP71 needs higher power
        w0=2.5e-6,  # 2.5μm束腰，LP71需要更大束腰 / 2.5μm beam waist, LP71 needs larger waist
        l=7  # LP71的轨道角动量量子数 / Orbital angular momentum quantum number for LP71
    )
    
    # 4. 设置LP71光场 / Setup LP71 optical field
    # 创建三维网格 / Create 3D grid
    x_range = np.linspace(-6e-6, 6e-6, 60)  # ±6μm，LP71需要更大范围 / ±6μm, LP71 needs larger range
    y_range = np.linspace(-6e-6, 6e-6, 60)
    z_range = np.linspace(-3e-6, 3e-6, 30)  # ±3μm
    
    def lp71_field_function(x, y, z):
        """LP71光场函数 / LP71 optical field function"""
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return create_lp71_field(r, phi, z, w0=optical_trap.w0, 
                               wavelength=optical_trap.wavelength, l=7, p=1)
    
    def lp71_phase_function(x, y, z):
        """LP71相位函数 / LP71 phase function"""
        phi = np.arctan2(y, x)
        return optical_trap.l * phi  # l=7的相位 / Phase for l=7
    
    optical_trap.set_field(x_range, y_range, z_range, 
                          lp71_field_function, lp71_phase_function)
    print("LP71光场设置完成 / LP71 optical field setup completed")
    
    # 5. 创建模拟盒子 / Create simulation box
    sim_box = SimulationBox(
        particles=particle,
        environment=environment,
        optical_trap=optical_trap
    )
    
    # 设置仿真参数 / Set simulation parameters
    sim_box.timestep = 5e-4  # 50μs时间步长，LP71需要更小时间步 / 50μs time step, LP71 needs smaller timestep
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
    duration = 3  # 30s模拟时间，LP71旋转更快 / 30s simulation time, LP71 rotates faster
    trajectory = sim_box.simulate(duration)
    
    print(f"模拟完成，总时间: {duration}s / Simulation completed, total time: {duration}s")
    print(f"时间步数: {len(trajectory[0]['time'])} / Number of time steps: {len(trajectory[0]['time'])}")
    
    # 7. 保存结果 / Save results
    output_file = "particle_trajectory_lp71_test.csv"
    sim_box.save_trajectory_to_csv(output_file)
    print(f"轨迹数据已保存到: {output_file} / Trajectory data saved to: {output_file}")
    
    # 8. 输出一些统计信息 / Output some statistics
    final_position = trajectory[0]['position'][-1]
    max_displacement = np.max(np.linalg.norm(trajectory[0]['position'], axis=1))
    mean_speed = np.mean(np.linalg.norm(trajectory[0]['velocity'], axis=1))
    max_force = np.max(np.linalg.norm(trajectory[0]['force'], axis=1))
    
    print("\n=== 模拟结果统计 / Simulation Results Statistics ===")
    print(f"最终位置 / Final position: ({final_position[0]*1e6:.2f}, {final_position[1]*1e6:.2f}, {final_position[2]*1e6:.2f}) μm")
    print(f"最大位移 / Maximum displacement: {max_displacement*1e6:.2f} μm")
    print(f"平均速度 / Average velocity: {mean_speed*1e6:.2f} μm/s")
    print(f"最大受力 / Maximum force: {max_force*1e12:.2f} pN")
    
    # 检查是否有角运动 / Check for angular motion
    max_angular_velocity = np.max(np.linalg.norm(trajectory[0]['angular_velocity'], axis=1))
    max_torque = np.max(np.linalg.norm(trajectory[0]['torque'], axis=1))
    print(f"最大角速度 / Maximum angular velocity: {max_angular_velocity:.2e} rad/s")
    print(f"最大扭矩 / Maximum torque: {max_torque*1e15:.2f} fN·m")
    
    # 新增：计算模拟用时、旋转角度和角速度统计 / New: Calculate simulation time, rotation angle and angular velocity statistics
    actual_simulation_time = trajectory[0]['time'][-1] - trajectory[0]['time'][0]
    print(f"\n=== LP71旋转运动分析 / LP71 Rotational Motion Analysis ===")
    print(f"实际模拟用时 / Actual simulation time: {actual_simulation_time:.3f} s")
    
    # 计算绕中心的旋转角度 / Calculate rotation angle around center
    positions = trajectory[0]['position']
    # 计算每个时刻相对于中心的角度 / Calculate angle relative to center at each time step
    angles = np.arctan2(positions[:, 1], positions[:, 0])  # 使用atan2计算角度
    
    # 处理角度跳跃（从-π到π的跳跃）/ Handle angle wrapping (jumps from -π to π)
    angle_diff = np.diff(angles)
    angle_diff[angle_diff > np.pi] -= 2 * np.pi
    angle_diff[angle_diff < -np.pi] += 2 * np.pi
    
    # 计算累积旋转角度 / Calculate cumulative rotation angle
    cumulative_rotation = np.sum(angle_diff)
    total_rotation_degrees = np.abs(cumulative_rotation) * 180 / np.pi
    
    print(f"LP71绕中心总旋转角度 / LP71 total rotation angle around center: {total_rotation_degrees:.2f}°")
    print(f"LP71绕中心总旋转角度 / LP71 total rotation angle around center: {np.abs(cumulative_rotation):.4f} rad")
    
    # 计算平均绕轴角速度 / Calculate average angular velocity around axis
    if actual_simulation_time > 0:
        avg_angular_velocity_around_center = np.abs(cumulative_rotation) / actual_simulation_time
        print(f"LP71平均绕中心角速度 / LP71 average angular velocity around center: {avg_angular_velocity_around_center:.4f} rad/s")
        print(f"LP71平均绕中心角速度 / LP71 average angular velocity around center: {avg_angular_velocity_around_center * 180 / np.pi:.2f} °/s")
    else:
        print("模拟时间为零，无法计算角速度 / Simulation time is zero, cannot calculate angular velocity")
    
    # 计算粒子自身的平均角速度（绕自身轴）/ Calculate particle's intrinsic angular velocity (around its own axis)
    intrinsic_angular_velocities = np.linalg.norm(trajectory[0]['angular_velocity'], axis=1)
    mean_intrinsic_angular_velocity = np.mean(intrinsic_angular_velocities)
    print(f"LP71平均自转角速度 / LP71 average intrinsic angular velocity: {mean_intrinsic_angular_velocity:.4f} rad/s")
    print(f"LP71平均自转角速度 / LP71 average intrinsic angular velocity: {mean_intrinsic_angular_velocity * 180 / np.pi:.2f} °/s")
    
    # 计算旋转频率 / Calculate rotation frequency
    if avg_angular_velocity_around_center > 0:
        rotation_frequency = avg_angular_velocity_around_center / (2 * np.pi)
        print(f"LP71绕中心旋转频率 / LP71 rotation frequency around center: {rotation_frequency:.6f} Hz")
        if rotation_frequency > 0:
            rotation_period = 1 / rotation_frequency
            print(f"LP71绕中心旋转周期 / LP71 rotation period around center: {rotation_period:.2f} s")
    
    # LP71特有的分析 / LP71-specific analysis
    print(f"\n=== LP71光束特性分析 / LP71 Beam Characteristics Analysis ===")
    print(f"轨道角动量量子数 l = {optical_trap.l}")
    print(f"径向量子数 p = 1 (LP71模式)")
    print(f"理论角动量传递效率较高，适合高精度旋转控制 / High theoretical angular momentum transfer efficiency, suitable for high-precision rotation control")
    
    return trajectory, sim_box

# 主程序 / Main program
if __name__ == "__main__":
    # 运行测试并捕获返回值 / Run test and capture return values
    trajectory, sim_box = test_lp71_single_particle()
    
    # 创建可视化器并加载数据 / Create visualizer and load data
    from simulation.visualizer import TrajectoryVisualizer
    visualizer = TrajectoryVisualizer("particle_trajectory_lp71_test.csv")
    
    # 重新创建光阱对象用于可视化 / Recreate optical trap object for visualization
    optical_trap = OpticalTrap(
        kappa=[2e-7, 2e-7, 1e-7],
        center=np.array([0.0, 0.0, 0.0]),
        wavelength=1064e-9,
        laser_power=1.5,
        w0=2.5e-6,
        l=7
    )
    
    # 重新设置光场 / Reset optical field
    x_range = np.linspace(-6e-6, 6e-6, 60)
    y_range = np.linspace(-6e-6, 6e-6, 60)
    z_range = np.linspace(-3e-6, 3e-6, 30)
    
    def lp71_field_function(x, y, z):
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return create_lp71_field(r, phi, z, w0=optical_trap.w0, 
                               wavelength=optical_trap.wavelength, l=7, p=1)
    
    optical_trap.set_field(x_range, y_range, z_range, lp71_field_function)
    
    # 绘制带有光场背景的轨迹图，增加等高线层数以显示多环结构
    visualizer.plot_2d_trajectory_with_field('xy', optical_trap=optical_trap, 
                                        field_alpha=0.6, field_levels=50)
    visualizer.plot_2d_trajectory_with_field('xz', optical_trap=optical_trap, 
                                        field_alpha=0.6, field_levels=50)
    visualizer.plot_2d_trajectory_with_field('yz', optical_trap=optical_trap, 
                                        field_alpha=0.6, field_levels=50)
    
    print("\nLP71测试完成！/ LP71 Test completed!")
    print("可以使用visualizer.py中的TrajectoryVisualizer类来可视化结果。/ You can use the TrajectoryVisualizer class in visualizer.py to visualize the results.")
    print("\n=== 使用说明 / Usage Instructions ===")
    print("1. 运行此脚本将自动尝试加载final_intensity_LP71.csv文件")
    print("2. 如果CSV文件无法读取，将使用理论LP71光场模型")
    print("3. 生成的轨迹数据保存在particle_trajectory_lp71_test.csv")
    print("4. 可视化结果将显示粒子在LP71光场中的旋转运动")