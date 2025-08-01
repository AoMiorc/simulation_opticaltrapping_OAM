import numpy as np
import sys
import os

# 添加simulation模块到路径 / Add simulation module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

from particle import ParticleFactory
from environment import Environment
from trap import OpticalTrap
from box import SimulationBox

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

def test_lg01_single_particle():
    """测试单粒子在LG01光束中的运动 / Test single particle motion in LG01 beam"""
    
    print("开始LG01单粒子测试... / Starting LG01 single particle test...")
    
    # 1. 创建粒子 / Create particle
    particle = ParticleFactory.create_polystyrene_sphere(
        radius=500e-9,  # 500nm聚苯乙烯球 / 500nm polystyrene sphere
        position=np.array([0.5e-6, 0.0, 0.0])  # 初始位置稍微偏离中心 / Initial position slightly off center
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
        laser_power=10,  # 100mW
        w0=2e-6,  # 2μm束腰 / 2μm beam waist
        l=1  # LG01的轨道角动量量子数 / Orbital angular momentum quantum number for LG01
    )
    
    # 4. 设置LG01光场 / Setup LG01 optical field
    # 创建三维网格 / Create 3D grid
    x_range = np.linspace(-5e-6, 5e-6, 50)  # ±5μm
    y_range = np.linspace(-5e-6, 5e-6, 50)
    z_range = np.linspace(-2e-6, 2e-6, 20)  # ±2μm
    
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
    
    # 在模拟开始前添加调试输出
    test_position = [0.5e-6, 0.5e-6, 0]  # 测试位置
    poynting_vec = optical_trap.get_poynting_vector_at_position(test_position)
    angular_momentum = optical_trap.get_angular_momentum_at_position(test_position)
    torque = optical_trap.calculate_torque_at_position(test_position)
    
    print(f"测试位置 {test_position}:")
    print(f"Poynting矢量: {poynting_vec}")
    print(f"角动量密度: {angular_momentum}")
    print(f"扭矩: {torque}")
    
    # 5. 创建模拟盒子 / Create simulation box
    sim_box = SimulationBox(
        particles=particle,
        environment=environment,
        optical_trap=optical_trap
    )
    
    # 设置仿真参数 / Set simulation parameters
    sim_box.timestep = 1e-3  # 1μs时间步长 / 1μs time step
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
    duration = 240  # 10ms模拟时间 / 10ms simulation time
    trajectory = sim_box.simulate(duration)
    
    print(f"模拟完成，总时间: {duration*1000}ms / Simulation completed, total time: {duration*1000}ms")
    print(f"时间步数: {len(trajectory[0]['time'])} / Number of time steps: {len(trajectory[0]['time'])}")
    
    # 7. 保存结果 / Save results
    output_file = "particle_trajectory_lg01_test.csv"
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
    print(f"\n=== 旋转运动分析 / Rotational Motion Analysis ===")
    print(f"实际模拟用时 / Actual simulation time: {actual_simulation_time*1000:.3f} ms")
    
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
    
    print(f"绕中心总旋转角度 / Total rotation angle around center: {total_rotation_degrees:.2f}°")
    print(f"绕中心总旋转角度 / Total rotation angle around center: {np.abs(cumulative_rotation):.4f} rad")
    
    # 计算平均绕轴角速度 / Calculate average angular velocity around axis
    if actual_simulation_time > 0:
        avg_angular_velocity_around_center = np.abs(cumulative_rotation) / actual_simulation_time
        print(f"平均绕中心角速度 / Average angular velocity around center: {avg_angular_velocity_around_center:.4f} rad/s")
        print(f"平均绕中心角速度 / Average angular velocity around center: {avg_angular_velocity_around_center * 180 / np.pi:.2f} °/s")
    else:
        print("模拟时间为零，无法计算角速度 / Simulation time is zero, cannot calculate angular velocity")
    
    # 计算粒子自身的平均角速度（绕自身轴）/ Calculate particle's intrinsic angular velocity (around its own axis)
    intrinsic_angular_velocities = np.linalg.norm(trajectory[0]['angular_velocity'], axis=1)
    mean_intrinsic_angular_velocity = np.mean(intrinsic_angular_velocities)
    print(f"平均自转角速度 / Average intrinsic angular velocity: {mean_intrinsic_angular_velocity:.4f} rad/s")
    print(f"平均自转角速度 / Average intrinsic angular velocity: {mean_intrinsic_angular_velocity * 180 / np.pi:.2f} °/s")
    
    # 计算旋转频率 / Calculate rotation frequency
    if avg_angular_velocity_around_center > 0:
        rotation_frequency = avg_angular_velocity_around_center / (2 * np.pi)
        print(f"绕中心旋转频率 / Rotation frequency around center: {rotation_frequency:.6f} Hz")
        if rotation_frequency > 0:
            rotation_period = 1 / rotation_frequency
            print(f"绕中心旋转周期 / Rotation period around center: {rotation_period*1000:.2f} ms")
    return trajectory, sim_box

# 移除函数外部的调试代码（第202-210行）
# 在test_lg01_single_particle.py的末尾添加 / Add at the end of test_lg01_single_particle.py
if __name__ == "__main__":
    # 运行测试并捕获返回值 / Run test and capture return values
    trajectory, sim_box = test_lg01_single_particle()
    
    # 创建可视化器并加载数据 / Create visualizer and load data
    from simulation.visualizer import TrajectoryVisualizer
    visualizer = TrajectoryVisualizer("particle_trajectory_lg01_test.csv")
    
    # 重新创建光阱对象用于可视化 / Recreate optical trap object for visualization
    optical_trap = OpticalTrap(
        kappa=[1e-6, 1e-6, 1e-7],
        center=np.array([0.0, 0.0, 0.0]),
        wavelength=1064e-9,
        laser_power=0.1,
        w0=2e-6,
        l=1
    )
    
    # 重新设置光场 / Reset optical field
    x_range = np.linspace(-5e-6, 5e-6, 50)
    y_range = np.linspace(-5e-6, 5e-6, 50)
    z_range = np.linspace(-2e-6, 2e-6, 20)
    
    def lg01_field_function(x, y, z):
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return create_lg01_field(r, phi, z, w0=optical_trap.w0, 
                               wavelength=optical_trap.wavelength, l=1, p=0)
    
    optical_trap.set_field(x_range, y_range, z_range, lg01_field_function)
    
    # 绘制带有光场背景的轨迹图 / Plot trajectory with optical field background
    visualizer.plot_2d_trajectory_with_field('xy', optical_trap=optical_trap, field_alpha=0.6)
    
    print("\n测试完成！/ Test completed!")
    print("可以使用visualizer.py中的TrajectoryVisualizer类来可视化结果。/ You can use the TrajectoryVisualizer class in visualizer.py to visualize the results.")