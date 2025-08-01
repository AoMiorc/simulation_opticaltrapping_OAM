import os
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# 添加simulation模块到路径 / Add simulation module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

from particle import ParticleFactory
from environment import Environment
from trap import OpticalTrap
from box import SimulationBox

def load_and_setup_csv_field(csv_filename, x_range, y_range, z_range):
    """
    从CSV文件加载光场数据并设置插值函数
    Load optical field data from CSV file and setup interpolation function
    """
    try:
        # 加载CSV数据
        intensity_data = np.loadtxt(csv_filename, delimiter=',')
        print(f"成功加载LP31光场数据，数据形状: {intensity_data.shape}")
        
        # 假设CSV数据是2D的强度分布（对应XY平面）
        # 如果数据格式不同，需要相应调整
        
        # 创建对应的坐标网格
        if len(intensity_data.shape) == 2:
            # 2D数据，假设对应XY平面
            ny, nx = intensity_data.shape
            
            # 创建对应的坐标
            x_csv = np.linspace(x_range[0], x_range[-1], nx)
            y_csv = np.linspace(y_range[0], y_range[-1], ny)
            
            # 创建3D数据（在Z方向复制）
            nz = len(z_range)
            intensity_3d = np.zeros((nx, ny, nz))
            
            for k in range(nz):
                # 在Z方向上，可以添加高斯衰减或保持不变
                z_factor = np.exp(-(z_range[k]**2) / (2 * (1e-6)**2))  # 1μm的Z方向衰减
                intensity_3d[:, :, k] = intensity_data.T * z_factor  # 转置以匹配坐标
            
            # 创建插值函数
            interpolator = RegularGridInterpolator(
                (x_csv, y_csv, z_range), 
                intensity_3d, 
                method='linear', 
                bounds_error=False, 
                fill_value=0.0
            )
            
            def csv_field_function(x, y, z):
                """基于CSV数据的LP31光场函数"""
                # 将输入转换为插值器需要的格式
                points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
                result = interpolator(points)
                return result.reshape(x.shape)
            
            return csv_field_function, intensity_data
            
        else:
            print(f"不支持的数据格式: {intensity_data.shape}")
            return None, None
            
    except Exception as e:
        print(f"加载CSV文件失败: {e}")
        return None, None

def test_lp31_with_csv_field():
    """使用CSV光场数据测试LP31单粒子运动"""
    
    print("开始LP31 CSV光场测试...")
    
    # 1. 创建粒子
    particle = ParticleFactory.create_polystyrene_sphere(
        radius=500e-9,  # 500nm聚苯乙烯球
        position=np.array([7e-6, 0.0, 0.0])  # 初始位置稍微偏离中心
    )
    print(f"创建粒子: 半径={particle.radius*1e9:.1f}nm, 质量={particle.mass*1e15:.2f}fg")
    
    # 2. 创建环境（水环境）
    environment = Environment(
        medium='liquid',
        T=298.0,  # 室温
        eta=0.001  # 水的粘度
    )
    print(f"环境设置: {environment.medium}, T={environment.T}K")
    
    # 3. 创建LP31光阱
    optical_trap = OpticalTrap(
        kappa=[2e-7, 2e-7, 1e-7],  # 阱刚度 [N/m]
        center=np.array([0.0, 0.0, 0.0]),
        wavelength=1064e-9,  # 1064nm激光
        laser_power=1.2,  # 120mW，LP31需要适中功率
        w0=2.0e-6,  # 2.0μm束腰
        l=3  # LP31的轨道角动量量子数
    )
    
    # 4. 设置网格范围（与LP31理论版本相同）
    x_range = np.linspace(-5e-6, 5e-6, 50)  # ±5μm
    y_range = np.linspace(-5e-6, 5e-6, 50)
    z_range = np.linspace(-2.5e-6, 2.5e-6, 25)  # ±2.5μm
    
    # 5. 加载CSV光场数据
    csv_filename = "final_intensity_LP31_2cm.csv"
    csv_field_function, intensity_data = load_and_setup_csv_field(
        csv_filename, x_range, y_range, z_range
    )
    
    if csv_field_function is None:
        print("无法加载CSV数据，程序退出")
        return None, None
    
    # 6. 设置相位函数（使用LP31的理论相位）
    def lp31_phase_function(x, y, z):
        """LP31相位函数"""
        phi = np.arctan2(y, x)
        return optical_trap.l * phi  # l=3的相位
    
    # 7. 设置光场
    optical_trap.set_field(x_range, y_range, z_range, 
                          csv_field_function, lp31_phase_function)
    print("LP31 CSV光场设置完成")
    
    # 8. 创建模拟盒子
    sim_box = SimulationBox(
        particles=particle,
        environment=environment,
        optical_trap=optical_trap
    )
    
    # 设置仿真参数（与LP71相同）
    sim_box.timestep = 5e-4  # 50μs时间步长
    sim_box.time = 0.0
    
    # 初始化阻尼系数
    sim_box.gamma = np.array([
        environment.get_drag_coefficient(particle),
        environment.get_drag_coefficient(particle),
        environment.get_drag_coefficient(particle)
    ])
    
    print(f"阻尼系数: γ={sim_box.gamma[0]:.2e} kg/s")
    
    # 9. 运行模拟（与LP71相同的总时长）
    print("开始模拟...")
    duration = 3  # 30秒模拟时间
    trajectory = sim_box.simulate(duration)
    
    print(f"模拟完成，总时间: {duration}s")
    print(f"时间步数: {len(trajectory[0]['time'])}")
    
    # 10. 保存结果
    output_file = "particle_trajectory_lp31_csv.csv"
    sim_box.save_trajectory_to_csv(output_file)
    print(f"轨迹数据已保存到: {output_file}")
    
    # 11. 输出统计信息
    final_position = trajectory[0]['position'][-1]
    max_displacement = np.max(np.linalg.norm(trajectory[0]['position'], axis=1))
    mean_speed = np.mean(np.linalg.norm(trajectory[0]['velocity'], axis=1))
    max_force = np.max(np.linalg.norm(trajectory[0]['force'], axis=1))
    
    print("\n=== 使用CSV光场的LP31模拟结果 ===")
    print(f"最终位置: ({final_position[0]*1e6:.2f}, {final_position[1]*1e6:.2f}, {final_position[2]*1e6:.2f}) μm")
    print(f"最大位移: {max_displacement*1e6:.2f} μm")
    print(f"平均速度: {mean_speed*1e6:.2f} μm/s")
    print(f"最大受力: {max_force*1e12:.2f} pN")
    
    # 检查是否有角运动
    max_angular_velocity = np.max(np.linalg.norm(trajectory[0]['angular_velocity'], axis=1))
    max_torque = np.max(np.linalg.norm(trajectory[0]['torque'], axis=1))
    print(f"最大角速度: {max_angular_velocity:.2e} rad/s")
    print(f"最大扭矩: {max_torque*1e15:.2f} fN·m")
    
    # 计算旋转运动分析
    actual_simulation_time = trajectory[0]['time'][-1] - trajectory[0]['time'][0]
    print(f"\n=== LP31旋转运动分析（基于CSV数据）===")
    print(f"实际模拟用时: {actual_simulation_time:.3f} s")
    
    # 计算绕中心的旋转角度
    positions = trajectory[0]['position']
    angles = np.arctan2(positions[:, 1], positions[:, 0])
    
    # 处理角度跳跃
    angle_diff = np.diff(angles)
    angle_diff[angle_diff > np.pi] -= 2 * np.pi
    angle_diff[angle_diff < -np.pi] += 2 * np.pi
    
    # 计算累积旋转角度
    cumulative_rotation = np.sum(angle_diff)
    total_rotation_degrees = np.abs(cumulative_rotation) * 180 / np.pi
    
    print(f"LP31绕中心总旋转角度: {total_rotation_degrees:.2f}°")
    print(f"LP31绕中心总旋转角度: {np.abs(cumulative_rotation):.4f} rad")
    
    # 计算平均绕轴角速度
    if actual_simulation_time > 0:
        avg_angular_velocity_around_center = np.abs(cumulative_rotation) / actual_simulation_time
        print(f"LP31平均绕中心角速度: {avg_angular_velocity_around_center:.4f} rad/s")
        print(f"LP31平均绕中心角速度: {avg_angular_velocity_around_center * 180 / np.pi:.2f} °/s")
        
        # 计算旋转频率
        if avg_angular_velocity_around_center > 0:
            rotation_frequency = avg_angular_velocity_around_center / (2 * np.pi)
            print(f"LP31绕中心旋转频率: {rotation_frequency:.6f} Hz")
            if rotation_frequency > 0:
                rotation_period = 1 / rotation_frequency
                print(f"LP31绕中心旋转周期: {rotation_period:.2f} s")
    
    # 计算粒子自身的平均角速度
    intrinsic_angular_velocities = np.linalg.norm(trajectory[0]['angular_velocity'], axis=1)
    mean_intrinsic_angular_velocity = np.mean(intrinsic_angular_velocities)
    print(f"LP31平均自转角速度: {mean_intrinsic_angular_velocity:.4f} rad/s")
    
    print(f"\n=== LP31 CSV光场特性分析 ===")
    print(f"轨道角动量量子数 l = {optical_trap.l}")
    print(f"径向量子数 p = 1 (LP31模式)")
    print(f"使用真实CSV光场数据，包含实际的强度分布和多环结构")
    print(f"相比理论模型，CSV数据可能包含更复杂的光场细节")
    
    return trajectory, sim_box

# 主程序
if __name__ == "__main__":
    trajectory, sim_box = test_lp31_with_csv_field()
    
    if trajectory is not None:
        # 可视化
        from simulation.visualizer import TrajectoryVisualizer
        visualizer = TrajectoryVisualizer("particle_trajectory_lp31_csv.csv")
        
        # 重新创建光阱用于可视化
        optical_trap = OpticalTrap(
            kappa=[2e-7, 2e-7, 1e-7],
            center=np.array([0.0, 0.0, 0.0]),
            wavelength=1064e-9,
            laser_power=1200,
            w0=2000.0e-6,
            l=3
        )
        
        # 重新设置CSV光场
        x_range = np.linspace(-5e-6, 5e-6, 50)
        y_range = np.linspace(-5e-6, 5e-6, 50)
        z_range = np.linspace(-2.5e-6, 2.5e-6, 25)
        
        csv_field_function, _ = load_and_setup_csv_field(
            "final_intensity_LP31_2cm.csv", x_range, y_range, z_range
        )
        
        if csv_field_function is not None:
            optical_trap.set_field(x_range, y_range, z_range, csv_field_function)
            
            # 绘制轨迹图，使用更多等高线层数显示多环结构
            visualizer.plot_2d_trajectory_with_field('xy', 
                                                    optical_trap=optical_trap, 
                                                    field_alpha=0.6, 
                                                    field_levels=50)
        
        print("\nLP31 CSV光场测试完成！")
        print("\n=== 使用说明 ===")
        print("1. 此脚本直接使用final_intensity_LP31_2cm.csv文件中的光场数据")
        print("2. 不依赖理论LP31光场模型，完全基于实验或计算得到的CSV数据")
        print("3. 生成的轨迹数据保存在particle_trajectory_lp31_csv.csv")
        print("4. 可视化结果将显示粒子在真实LP31光场中的运动")
        print("5. 使用与LP71相同的时间步长(50μs)和总时长(30s)设置")
        print("6. 相位函数仍使用理论模型(l=3)，强度分布完全来自CSV数据")