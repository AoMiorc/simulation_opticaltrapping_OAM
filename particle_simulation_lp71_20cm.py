import os
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# 添加simulation模块到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

from particle import ParticleFactory
from environment import Environment
from trap import OpticalTrap
from box import SimulationBox

def load_and_setup_csv_field(csv_filename, field_size_cm=2.0, z_decay_factor=0.5):
    """
    从CSV文件加载LP71光场数据并设置插值函数
    
    Args:
        csv_filename: CSV文件路径
        field_size_cm: 光场尺寸（厘米），默认2cm x 2cm
        z_decay_factor: Z方向衰减因子
    """
    try:
        print(f"正在加载LP71光场数据: {csv_filename}")
        
        # 检查文件是否存在
        if not os.path.exists(csv_filename):
            raise FileNotFoundError(f"找不到CSV文件: {csv_filename}")
        
        # 加载CSV数据
        intensity_data = np.loadtxt(csv_filename, delimiter=',')
        print(f"成功加载光场数据，数据形状: {intensity_data.shape}")
        
        # 假设CSV数据是正方形的2D强度分布
        if len(intensity_data.shape) == 2:
            ny, nx = intensity_data.shape
            
            # 创建坐标网格（转换为米）
            field_size_m = field_size_cm * 0.01  # 转换为米
            x_coords = np.linspace(-field_size_m/2, field_size_m/2, nx)
            y_coords = np.linspace(-field_size_m/2, field_size_m/2, ny)
            
            # 创建Z方向坐标（较小范围）
            z_coords = np.linspace(-field_size_m/10, field_size_m/10, 20)  # Z方向范围较小
            
            # 创建3D数据（在Z方向添加衰减）
            nz = len(z_coords)
            intensity_3d = np.zeros((nx, ny, nz))
            
            for k, z in enumerate(z_coords):
                # Z方向高斯衰减
                z_factor = np.exp(-z_decay_factor * (z**2) / (field_size_m/10)**2)
                intensity_3d[:, :, k] = intensity_data.T * z_factor
            
            # 创建插值函数
            interpolator = RegularGridInterpolator(
                (x_coords, y_coords, z_coords), 
                intensity_3d, 
                method='linear', 
                bounds_error=False, 
                fill_value=0.0
            )
            
            def csv_field_function(x, y, z):
                """基于CSV数据的LP71光场函数"""
                # 确保输入是numpy数组
                x = np.asarray(x)
                y = np.asarray(y)
                z = np.asarray(z)
                
                # 将输入转换为插值器需要的格式
                points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
                result = interpolator(points)
                return result.reshape(x.shape)
            
            print(f"光场范围: X=[{-field_size_m/2*1e6:.1f}, {field_size_m/2*1e6:.1f}] μm")
            print(f"光场范围: Y=[{-field_size_m/2*1e6:.1f}, {field_size_m/2*1e6:.1f}] μm")
            print(f"光场范围: Z=[{-field_size_m/10*1e6:.1f}, {field_size_m/10*1e6:.1f}] μm")
            
            return csv_field_function, (x_coords, y_coords, z_coords), intensity_data
            
        else:
            raise ValueError(f"不支持的数据格式: {intensity_data.shape}")
            
    except Exception as e:
        print(f"加载CSV文件失败: {e}")
        return None, None, None

def simulate_particle_in_lp71_field():
    """
    在LP71 CSV光场中模拟粒子运动
    """
    print("\n=== LP71 CSV光场粒子运动模拟 ===")
    
    # 1. 创建粒子（聚苯乙烯微球）
    particle = ParticleFactory.create_polystyrene_sphere(
        radius=500e-9,  # 500nm半径
        position=np.array([1e-6, 0.5e-6, 0.0])  # 初始位置稍微偏离中心
    )
    print(f"粒子参数: 半径={particle.radius*1e9:.1f}nm, 质量={particle.mass*1e15:.2f}fg")
    
    # 2. 创建环境（水环境）
    environment = Environment(
        medium='liquid',
        T=298.0,  # 室温
        eta=0.001  # 水的粘度
    )
    print(f"环境设置: {environment.medium}, T={environment.T}K, η={environment.eta} Pa·s")
    
    # 3. 加载CSV光场数据
    csv_filename = "final_intensity_LP71_minus6_20cm.csv"
    csv_field_function, coords, intensity_data = load_and_setup_csv_field(
        csv_filename, field_size_cm=2.0, z_decay_factor=0.5
    )
    
    if csv_field_function is None:
        print("无法加载CSV数据，程序退出")
        return None, None
    
    x_coords, y_coords, z_coords = coords
    
    # 4. 创建光阱
    optical_trap = OpticalTrap(
        kappa=[1e-6, 1e-6, 5e-7],  # 弹簧常数
        center=np.array([0.0, 0.0, 0.0]),
        wavelength=1064e-9,
        laser_power=2.0,  # 激光功率
        w0=3e-6,  # 束腰半径
        l=7  # 轨道角动量量子数
    )
    
    # 5. 设置LP71相位函数
    def lp71_phase_function(x, y, z):
        """LP71模式的相位函数"""
        phi = np.arctan2(y, x)
        return optical_trap.l * phi
    
    # 6. 设置光场
    optical_trap.set_field(x_coords, y_coords, z_coords, 
                          csv_field_function, lp71_phase_function)
    print("CSV光场设置完成")
    
    # 7. 创建模拟盒子
    sim_box = SimulationBox(
        particles=particle,
        environment=environment,
        optical_trap=optical_trap,
        timestep=5e-3  # 1ms时间步长
    )
    
    # 8. 运行模拟
    print("\n开始粒子运动模拟...")
    duration = 30.0  # 模拟20秒
    
    try:
        trajectory = sim_box.simulate(duration)
        print(f"模拟完成，总时长: {duration}s")
        
        # 9. 保存轨迹数据
        output_file = "particle_trajectory_lp71_20cm.csv"
        sim_box.save_trajectory_to_csv(output_file)
        print(f"轨迹数据已保存到: {output_file}")
        
        # 10. 分析结果
        analyze_trajectory(trajectory)
        
        return trajectory, sim_box, optical_trap
        
    except Exception as e:
        print(f"模拟过程中出现错误: {e}")
        return None, None, None

def analyze_trajectory(trajectory):
    """
    分析粒子轨迹
    """
    if not trajectory or len(trajectory) == 0:
        print("没有轨迹数据可分析")
        return
    
    particle_data = trajectory[0]
    positions = np.array(particle_data['position'])
    velocities = np.array(particle_data['velocity'])
    times = np.array(particle_data['time'])
    
    # 计算统计量
    initial_position = positions[0]
    final_position = positions[-1]
    displacement = final_position - initial_position
    max_displacement = np.max(np.linalg.norm(positions - initial_position, axis=1))
    mean_speed = np.mean(np.linalg.norm(velocities, axis=1))
    max_speed = np.max(np.linalg.norm(velocities, axis=1))
    
    # 计算径向距离
    radial_distances = np.linalg.norm(positions[:, :2], axis=1)  # XY平面径向距离
    mean_radial_distance = np.mean(radial_distances)
    max_radial_distance = np.max(radial_distances)
    
    print("\n=== 轨迹分析结果 ===")
    print(f"初始位置: ({initial_position[0]*1e6:.2f}, {initial_position[1]*1e6:.2f}, {initial_position[2]*1e6:.2f}) μm")
    print(f"最终位置: ({final_position[0]*1e6:.2f}, {final_position[1]*1e6:.2f}, {final_position[2]*1e6:.2f}) μm")
    print(f"总位移: ({displacement[0]*1e6:.2f}, {displacement[1]*1e6:.2f}, {displacement[2]*1e6:.2f}) μm")
    print(f"最大位移: {max_displacement*1e6:.2f} μm")
    print(f"平均速度: {mean_speed*1e6:.2f} μm/s")
    print(f"最大速度: {max_speed*1e6:.2f} μm/s")
    print(f"平均径向距离: {mean_radial_distance*1e6:.2f} μm")
    print(f"最大径向距离: {max_radial_distance*1e6:.2f} μm")
    print(f"模拟时长: {times[-1]:.2f} s")
    print(f"数据点数: {len(times)}")

def visualize_results():
    """
    可视化模拟结果
    """
    try:
        from simulation.visualizer import TrajectoryVisualizer
        
        # 检查轨迹文件是否存在
        trajectory_file = "particle_trajectory_lp71_20cm.csv"
        if not os.path.exists(trajectory_file):
            print(f"轨迹文件 {trajectory_file} 不存在，请先运行模拟")
            return
        
        print("\n开始可视化...")
        visualizer = TrajectoryVisualizer(trajectory_file)
        
        # 重新加载光场用于可视化
        csv_field_function, coords, _ = load_and_setup_csv_field(
            "final_intensity_LP71_minus6_20cm.csv", field_size_cm=2.0
        )
        
        if csv_field_function is not None:
            x_coords, y_coords, z_coords = coords
            
            # 创建光阱用于可视化
            optical_trap = OpticalTrap(
                kappa=[1e-6, 1e-6, 5e-7],
                center=np.array([0.0, 0.0, 0.0]),
                wavelength=1064e-9,
                laser_power=2.0,
                w0=3e-6,
                l=7
            )
            
            def lp71_phase_function(x, y, z):
                phi = np.arctan2(y, x)
                return optical_trap.l * phi
            
            optical_trap.set_field(x_coords, y_coords, z_coords, 
                                  csv_field_function, lp71_phase_function)
            
            # 绘制2D轨迹图
            print("绘制XY平面轨迹图...")
            visualizer.plot_2d_trajectory_with_field('xy', 
                                                    optical_trap=optical_trap, 
                                                    field_alpha=0.7, 
                                                    field_levels=30)
            
            # 绘制3D轨迹图
            print("绘制3D轨迹图...")
            visualizer.plot_3d_trajectory(show_field=True, optical_trap=optical_trap)
            
            print("可视化完成！")
        else:
            print("无法加载光场数据进行可视化")
            
    except ImportError:
        print("无法导入可视化模块，跳过可视化")
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")

if __name__ == "__main__":
    print("LP71光场粒子运动模拟程序")
    print("使用CSV文件: final_intensity_LP71_minus6_20cm.csv")
    
    # 运行模拟
    trajectory, sim_box, optical_trap = simulate_particle_in_lp71_field()
    
    if trajectory is not None:
        print("\n模拟成功完成！")
        
        # 可视化结果
        visualize_results()
        
        print("\n程序执行完毕。")
        print("生成的文件:")
        print("- particle_trajectory_lp71_20cm.csv: 粒子轨迹数据")
        print("- 各种可视化图表")
    else:
        print("\n模拟失败，请检查输入文件和参数设置。")