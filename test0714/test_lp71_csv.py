import os
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# 添加simulation模块到路径 / Add simulation module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simulation'))

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
        print(f"成功加载光场数据，数据形状: {intensity_data.shape}")
        
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
                """基于CSV数据的光场函数"""
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

def test_lp71_with_csv_field():
    """使用CSV光场数据测试单粒子运动"""
    
    print("开始LP71 CSV光场测试...")
    
    # 1. 创建粒子
    particle = ParticleFactory.create_polystyrene_sphere(
        radius=500e-9,
        position=np.array([6e-6, 0.0, 0.0])  # 初始位置
    )
    print(f"创建粒子: 半径={particle.radius*1e9:.1f}nm")
    
    # 2. 创建环境
    environment = Environment(
        medium='liquid',
        T=297.0,
        eta=0.001
    )
    
    # 3. 创建光阱
    optical_trap = OpticalTrap(
        kappa=[2e-7, 2e-7, 1e-7],
        center=np.array([0.0, 0.0, 0.0]),
        wavelength=1064e-9,
        laser_power=1.5,
        w0=2.5e-6,
        l=-6
    )
    
    # 4. 设置网格范围 - 增加分辨率
    x_range = np.linspace(-6e-6, 6e-6, 300)  # 从60增加到300
    y_range = np.linspace(-6e-6, 6e-6, 300)  # 从60增加到300
    z_range = np.linspace(-3e-6, 3e-6, 150)  # 从30增加到150
    
    # 5. 加载CSV光场数据 - 修复文件名和路径
    csv_filename = os.path.join(os.path.dirname(__file__), "final_intensity_LP71_minus6_20cm 1.csv")
    csv_field_function, intensity_data = load_and_setup_csv_field(
        csv_filename, x_range, y_range, z_range
    )
    
    if csv_field_function is None:
        print("无法加载CSV数据，程序退出")
        return None, None
    
    # 6. 设置相位函数（仍然使用理论相位）
    def lp71_phase_function(x, y, z):
        phi = np.arctan2(y, x)
        return optical_trap.l * phi
    
    # 7. 设置光场
    optical_trap.set_field(x_range, y_range, z_range, 
                          csv_field_function, lp71_phase_function)
    print("CSV光场设置完成")
    
    # 8. 创建模拟盒子
    sim_box = SimulationBox(
        particles=particle,
        environment=environment,
        optical_trap=optical_trap
    )
    
    # 设置仿真参数
    sim_box.timestep = 5e-4  # 50μs
    sim_box.time = 0.0
    
    # 初始化阻尼系数
    sim_box.gamma = np.array([
        environment.get_drag_coefficient(particle),
        environment.get_drag_coefficient(particle),
        environment.get_drag_coefficient(particle)
    ])
    
    # 9. 运行模拟
    print("开始模拟...")
    duration = 0.3  # 30秒
    trajectory = sim_box.simulate(duration)
    
    # 10. 保存结果
    output_file = "particle_trajectory_lp71_csv.csv"
    sim_box.save_trajectory_to_csv(output_file)
    print(f"轨迹数据已保存到: {output_file}")
    
    # 11. 输出统计信息
    final_position = trajectory[0]['position'][-1]
    max_displacement = np.max(np.linalg.norm(trajectory[0]['position'], axis=1))
    mean_speed = np.mean(np.linalg.norm(trajectory[0]['velocity'], axis=1))
    
    print("\n=== 使用CSV光场的模拟结果 ===")
    print(f"最终位置: ({final_position[0]*1e6:.2f}, {final_position[1]*1e6:.2f}, {final_position[2]*1e6:.2f}) μm")
    print(f"最大位移: {max_displacement*1e6:.2f} μm")
    print(f"平均速度: {mean_speed*1e6:.2f} μm/s")
    
    return trajectory, sim_box

# 主程序
if __name__ == "__main__":
    trajectory, sim_box = test_lp71_with_csv_field()
    
    if trajectory is not None:
        # 可视化
        from visualizer import TrajectoryVisualizer
        import matplotlib.pyplot as plt
        
        visualizer = TrajectoryVisualizer("particle_trajectory_lp71_csv.csv")
        
        # 重新创建光阱用于可视化
        optical_trap = OpticalTrap(
            kappa=[2e-7, 2e-7, 1e-7],
            center=np.array([0.0, 0.0, 0.0]),
            wavelength=1064e-9,
            laser_power=1.5,
            w0=2.5e-6,
            l=-6
        )
        
        # 重新设置CSV光场 - 修复文件名
        x_range = np.linspace(-6e-6, 6e-6, 60)
        y_range = np.linspace(-6e-6, 6e-6, 60)
        z_range = np.linspace(-3e-6, 3e-6, 30)
        
        csv_field_function, _ = load_and_setup_csv_field(
            os.path.join(os.path.dirname(__file__), "final_intensity_LP71_minus6_20cm 1.csv"), 
            x_range, y_range, z_range
        )
        
        if csv_field_function is not None:
            # 添加相位函数
            def lp71_phase_function(x, y, z):
                phi = np.arctan2(y, x)
                return optical_trap.l * phi
            
            optical_trap.set_field(x_range, y_range, z_range, 
                                  csv_field_function, lp71_phase_function)
            
            # 绘制轨迹图，使用更多等高线层数显示多环结构
            visualizer.plot_2d_trajectory_with_point_field('xy', 
                                                    optical_trap=optical_trap, 
                                                    field_alpha=0.6)
            
            # 显示图形
            plt.show()
        
        print("\nLP71 CSV光场测试完成！")
