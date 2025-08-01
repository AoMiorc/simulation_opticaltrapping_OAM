import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator

# 导入必要的模块
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simulation'))
from particle import ParticleFactory
from environment import Environment
from trap import OpticalTrap
from box import SimulationBox

def load_and_setup_csv_field(intensity_csv, phase_csv, x_range, y_range, z_range):
    """
    从CSV文件加载光场强度和相位数据并设置插值函数
    Load optical field intensity and phase data from CSV files and setup interpolation function
    """
    try:
        # 加载强度和相位CSV数据
        intensity_data = np.loadtxt(intensity_csv, delimiter=',')
        phase_data = np.loadtxt(phase_csv, delimiter=',')
        print(f"成功加载强度数据，数据形状: {intensity_data.shape}")
        print(f"成功加载相位数据，数据形状: {phase_data.shape}")
        
        # 检查数据形状是否匹配
        if intensity_data.shape != phase_data.shape:
            print(f"警告：强度和相位数据形状不匹配！")
            return None, None, None, None
        
        # 假设CSV数据是2D的分布（对应XY平面）
        if len(intensity_data.shape) == 2:
            # 2D数据，假设对应XY平面
            ny, nx = intensity_data.shape
            
            # 创建对应的坐标
            x_csv = np.linspace(x_range[0], x_range[-1], nx)
            y_csv = np.linspace(y_range[0], y_range[-1], ny)
            
            # 创建3D数据（在Z方向复制）
            nz = len(z_range)
            intensity_3d = np.zeros((nx, ny, nz))
            phase_3d = np.zeros((nx, ny, nz))
            
            for k in range(nz):
                # 在Z方向上，可以添加高斯衰减或保持不变
                z_factor = np.exp(-(z_range[k]**2) / (2 * (1e-6)**2))  # 1μm的Z方向衰减
                intensity_3d[:, :, k] = intensity_data.T * z_factor  # 转置以匹配坐标
                phase_3d[:, :, k] = phase_data.T  # 相位不衰减
            
            # 创建插值函数
            intensity_interpolator = RegularGridInterpolator(
                (x_csv, y_csv, z_range), 
                intensity_3d, 
                method='linear', 
                bounds_error=False, 
                fill_value=0.0
            )
            
            phase_interpolator = RegularGridInterpolator(
                (x_csv, y_csv, z_range), 
                phase_3d, 
                method='linear', 
                bounds_error=False, 
                fill_value=0.0
            )
            
            def csv_intensity_function(x, y, z):
                """基于CSV数据的光场强度函数"""
                points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
                result = intensity_interpolator(points)
                return result.reshape(x.shape)
            
            def csv_phase_function(x, y, z):
                """基于CSV数据的光场相位函数"""
                points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
                result = phase_interpolator(points)
                return result.reshape(x.shape)
            
            return csv_intensity_function, csv_phase_function, intensity_data, phase_data
            
        else:
            print(f"不支持的数据格式: {intensity_data.shape}")
            return None, None, None, None
            
    except Exception as e:
        print(f"加载CSV文件失败: {e}")
        return None, None, None, None

def test_lp71_with_csv_field_new():
    """使用新的CSV光场数据测试单粒子运动"""
    
    print("开始LP71 新CSV光场测试...")
    
    # 1. 创建粒子
    particle = ParticleFactory.create_polystyrene_sphere(
        radius=500e-9,
        position=np.array([0.0, 0.1e-6, 0.0])  # 初始位置
    )
    print(f"创建粒子: 半径={particle.radius*1e9:.1f}nm")
    
    # 2. 创建环境
    environment = Environment(
        medium='liquid',
        T=0.0,
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
    
    # 5. 加载CSV光场数据 - 使用新的强度和相位文件
    intensity_csv = os.path.join(os.path.dirname(__file__), "final_intensity_LP71_m6_0cm.csv")
    phase_csv = os.path.join(os.path.dirname(__file__), "final_phase_LP71_m6_0cm.csv")
    
    csv_intensity_function, csv_phase_function, intensity_data, phase_data = load_and_setup_csv_field(
        intensity_csv, phase_csv, x_range, y_range, z_range
    )
    
    if csv_intensity_function is None or csv_phase_function is None:
        print("无法加载CSV数据，程序退出")
        return None, None
    
    # 6. 设置光场
    optical_trap.set_field(x_range, y_range, z_range, 
                          csv_intensity_function, csv_phase_function)
    print("新CSV光场设置完成")
    
    # 7. 创建模拟盒子
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
    
    # 8. 运行模拟
    print("开始模拟...")
    duration = 0.1  # 30秒
    trajectory = sim_box.simulate(duration)
    
    # 9. 保存结果
    output_file = "particle_trajectory_lp71_csv_new.csv"
    sim_box.save_trajectory_to_csv(output_file)
    print(f"轨迹数据已保存到: {output_file}")
    
    # 10. 输出统计信息
    final_position = trajectory[0]['position'][-1]
    max_displacement = np.max(np.linalg.norm(trajectory[0]['position'], axis=1))
    mean_speed = np.mean(np.linalg.norm(trajectory[0]['velocity'], axis=1))
    
    print("\n=== 使用新CSV光场的模拟结果 ===")
    print(f"最终位置: ({final_position[0]*1e6:.2f}, {final_position[1]*1e6:.2f}, {final_position[2]*1e6:.2f}) μm")
    print(f"最大位移: {max_displacement*1e6:.2f} μm")
    print(f"平均速度: {mean_speed*1e6:.2f} μm/s")
    
    return trajectory, sim_box

# 主程序
if __name__ == "__main__":
    trajectory, sim_box = test_lp71_with_csv_field_new()
    
    if trajectory is not None:
        # 可视化
        try:
            from visualizer import TrajectoryVisualizer
            import matplotlib.pyplot as plt
            
            visualizer = TrajectoryVisualizer("particle_trajectory_lp71_csv_new.csv")
            
            # 重新创建光阱用于可视化
            optical_trap = OpticalTrap(
                kappa=[2e-7, 2e-7, 1e-7],
                center=np.array([0.0, 0.0, 0.0]),
                wavelength=1064e-9,
                laser_power=1.5,
                w0=2.5e-6,
                l=-6
            )
            
            # 重新设置CSV光场 - 使用高分辨率
            x_range = np.linspace(-6e-6, 6e-6, 200)
            y_range = np.linspace(-6e-6, 6e-6, 200)
            z_range = np.linspace(-3e-6, 3e-6, 100)
            
            intensity_csv = os.path.join(os.path.dirname(__file__), "final_intensity_LP71_m6_0cm.csv")
            phase_csv = os.path.join(os.path.dirname(__file__), "final_phase_LP71_m6_0cm.csv")
            
            csv_intensity_function, csv_phase_function, _, _ = load_and_setup_csv_field(
                intensity_csv, phase_csv, x_range, y_range, z_range
            )
            
            if csv_intensity_function is not None and csv_phase_function is not None:
                optical_trap.set_field(x_range, y_range, z_range, 
                                      csv_intensity_function, csv_phase_function)
                
                # 绘制轨迹图，使用更多等高线层数显示多环结构
                visualizer.plot_2d_trajectory_with_point_field('xy', 
                                                        optical_trap=optical_trap, 
                                                        field_alpha=0.6)
                
                # 显示图形
                plt.show()
            
        except ImportError as e:
            print(f"可视化模块导入失败: {e}")
            print("跳过可视化步骤")
        
        print("\nLP71 新CSV光场测试完成！")


def load_and_setup_csv_field(intensity_file, phase_file, x_range, y_range, z_range):
    """
    从CSV文件加载光场强度和相位数据并设置插值函数
    Load optical field intensity and phase data from CSV files and setup interpolation function
    """
    try:
        # 加载强度和相位CSV数据
        intensity_data = np.loadtxt(intensity_file, delimiter=',')
        phase_data = np.loadtxt(phase_file, delimiter=',')
        print(f"成功加载强度数据，数据形状: {intensity_data.shape}")
        print(f"成功加载相位数据，数据形状: {phase_data.shape}")
        
        # **修改：不进行全局相位展开，而是保持原始相位结构**
        # 注释掉相位展开处理
        # phase_data = np.unwrap(phase_data, axis=0)  # 沿x方向展开
        # phase_data = np.unwrap(phase_data, axis=1)  # 沿y方向展开
        
        # **新增：检测并标记内外环区域**
        ny, nx = phase_data.shape
        center_x, center_y = nx // 2, ny // 2
        
        # 创建径向距离数组
        y_indices, x_indices = np.ogrid[:ny, :nx]
        r_grid = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
        
        # 定义内外环边界（调整此值，例如基于数据可视化）
        boundary_radius = 1.25e-6 / (x_csv[1] - x_csv[0])  # 转换为网格单位，假设1.5μm边界
        
        # 分别处理内外环的相位
        inner_mask = r_grid < boundary_radius
        outer_mask = r_grid >= boundary_radius
        
        # 只在各自区域内进行局部相位展开
        if np.any(inner_mask):
            inner_phase = phase_data.copy()
            inner_phase[~inner_mask] = 0
            inner_phase = np.unwrap(inner_phase, axis=0)
            inner_phase = np.unwrap(inner_phase, axis=1)
            phase_data[inner_mask] = inner_phase[inner_mask]
        
        if np.any(outer_mask):
            outer_phase = phase_data.copy()
            outer_phase[~outer_mask] = 0
            outer_phase = np.unwrap(outer_phase, axis=0)
            outer_phase = np.unwrap(outer_phase, axis=1)
            # **新增：对外环相位应用符号翻转，如果相位相反**
            outer_phase = -outer_phase  # 假设外环需要反转符号
            phase_data[outer_mask] = outer_phase[outer_mask]
        
        # 检查数据形状是否匹配
        if intensity_data.shape != phase_data.shape:
            print(f"警告：强度和相位数据形状不匹配！")
            return None, None, None, None
        
        # 假设CSV数据是2D的分布（对应XY平面）
        if len(intensity_data.shape) == 2:
            # 2D数据，假设对应XY平面
            ny, nx = intensity_data.shape
            
            # 创建对应的坐标
            x_csv = np.linspace(x_range[0], x_range[-1], nx)
            y_csv = np.linspace(y_range[0], y_range[-1], ny)
            
            # 创建3D数据（在Z方向复制）
            nz = len(z_range)
            intensity_3d = np.zeros((nx, ny, nz))
            phase_3d = np.zeros((nx, ny, nz))
            
            for k in range(nz):
                # 在Z方向上，可以添加高斯衰减或保持不变
                z_factor = np.exp(-(z_range[k]**2) / (2 * (1e-6)**2))  # 1μm的Z方向衰减
                intensity_3d[:, :, k] = intensity_data.T * z_factor  # 转置以匹配坐标
                phase_3d[:, :, k] = phase_data.T  # 相位不衰减
            
            # 创建插值函数
            intensity_interpolator = RegularGridInterpolator(
                (x_csv, y_csv, z_range), 
                intensity_3d, 
                method='linear', 
                bounds_error=False, 
                fill_value=0.0
            )
            
            phase_interpolator = RegularGridInterpolator(
                (x_csv, y_csv, z_range), 
                phase_3d, 
                method='linear', 
                bounds_error=False, 
                fill_value=0.0
            )
            
            def csv_intensity_function(x, y, z):
                """基于CSV数据的光场强度函数"""
                points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
                result = intensity_interpolator(points)
                return result.reshape(x.shape)
            
            def csv_phase_function(x, y, z):
                """基于CSV数据的光场相位函数"""
                points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
                result = phase_interpolator(points)
                return result.reshape(x.shape)
            
            return csv_intensity_function, csv_phase_function, intensity_data, phase_data
            
        else:
            print(f"不支持的数据格式: {intensity_data.shape}")
            return None, None, None, None
            
    except Exception as e:
        print(f"加载CSV文件失败: {e}")
        return None, None, None, None