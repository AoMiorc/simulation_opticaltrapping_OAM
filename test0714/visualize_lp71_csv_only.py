import os
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# 添加simulation模块到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simulation'))

from trap import OpticalTrap
from visualizer import TrajectoryVisualizer

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

def create_optical_trap_for_visualization(csv_filename, resolution='low'):
    """
    创建用于可视化的光阱对象
    
    Args:
        csv_filename: CSV光场数据文件名
        resolution: 分辨率设置 ('low', 'medium', 'high')
    """
    # 创建光阱对象
    optical_trap = OpticalTrap(
        kappa=[2e-7, 2e-7, 1e-7],
        center=np.array([0.0, 0.0, 0.0]),
        wavelength=1064e-9,
        laser_power=1.5,
        w0=2.5e-6,
        l=-7
    )
    
    # 根据分辨率设置网格
    if resolution == 'low':
        x_range = np.linspace(-6e-6, 6e-6, 60)
        y_range = np.linspace(-6e-6, 6e-6, 60)
        z_range = np.linspace(-3e-6, 3e-6, 30)
    elif resolution == 'medium':
        x_range = np.linspace(-6e-6, 6e-6, 120)
        y_range = np.linspace(-6e-6, 6e-6, 120)
        z_range = np.linspace(-3e-6, 3e-6, 60)
    elif resolution == 'high':
        x_range = np.linspace(-6e-6, 6e-6, 200)
        y_range = np.linspace(-6e-6, 6e-6, 200)
        z_range = np.linspace(-3e-6, 3e-6, 100)
    else:
        raise ValueError("分辨率必须是 'low', 'medium', 或 'high'")
    
    # 加载CSV光场数据
    csv_field_function, intensity_data = load_and_setup_csv_field(
        csv_filename, x_range, y_range, z_range
    )
    
    if csv_field_function is None:
        print("无法加载CSV数据")
        return None
    
    # 设置相位函数
    def lp71_phase_function(x, y, z):
        phi = np.arctan2(y, x)
        return optical_trap.l * phi
    
    # 设置光场
    optical_trap.set_field(x_range, y_range, z_range, 
                          csv_field_function, lp71_phase_function)
    
    print(f"光场设置完成，分辨率: {resolution}")
    return optical_trap

def visualize_lp71_trajectory(trajectory_csv="particle_trajectory_lp71_csv.csv", 
                             field_csv=None,  # 改为None，在函数内部设置
                             resolution='low',
                             plane='xy',
                             figsize=(12, 10),
                             field_alpha=0.6):
    """
    可视化LP71轨迹和光场
    """
    # 如果没有指定field_csv，使用默认路径
    if field_csv is None:
        field_csv = os.path.join(os.path.dirname(__file__), "final_intensity_LP71_m6_0cm.csv")
    
    print(f"开始可视化LP71轨迹...")
    print(f"轨迹文件: {trajectory_csv}")
    print(f"光场文件: {field_csv}")
    
    # 检查文件是否存在
    if not os.path.exists(trajectory_csv):
        print(f"错误: 轨迹文件 {trajectory_csv} 不存在")
        return
    
    if not os.path.exists(field_csv):
        print(f"错误: 光场文件 {field_csv} 不存在")
        return
    
    # 创建可视化器
    visualizer = TrajectoryVisualizer(trajectory_csv)
    
    # 创建光阱对象
    optical_trap = create_optical_trap_for_visualization(field_csv, resolution)
    
    if optical_trap is None:
        print("无法创建光阱对象，仅绘制轨迹")
        visualizer.plot_2d_trajectory(plane, figsize)
        return
    
    # 绘制带光场的轨迹图
    visualizer.plot_2d_trajectory_with_point_field(
        plane=plane, 
        figsize=figsize,
        optical_trap=optical_trap, 
        field_alpha=field_alpha
    )
    
    print("可视化完成！")

def visualize_multiple_planes(trajectory_csv="particle_trajectory_lp71_csv.csv", 
                             field_csv=None,  # 改为None
                             resolution='low',
                             field_alpha=0.6):
    """
    在多个平面上可视化轨迹
    """
    planes = ['xy', 'xz', 'yz']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 创建可视化器
    visualizer = TrajectoryVisualizer(trajectory_csv)
    
    # 创建光阱对象
    optical_trap = create_optical_trap_for_visualization(field_csv, resolution)
    
    for i, plane in enumerate(planes):
        plt.subplot(1, 3, i+1)
        
        if optical_trap is not None:
            # 这里需要修改visualizer来支持指定的axes
            visualizer.plot_2d_trajectory_with_point_field(
                plane=plane,
                optical_trap=optical_trap, 
                field_alpha=field_alpha
            )
        else:
            visualizer.plot_2d_trajectory(plane)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数 - 可视化示例
    """
    print("=== LP71 CSV光场可视化工具 ===")
    
    # 默认参数
    trajectory_file = "particle_trajectory_lp71_csv_new.csv"
    field_file = os.path.join(os.path.dirname(__file__), "final_intensity_LP71_m6_0cm.csv")
    
    # 检查文件是否存在
    if not os.path.exists(trajectory_file):
        print(f"警告: 轨迹文件 {trajectory_file} 不存在")
        print("请先运行 test_lp71_csv.py 生成轨迹数据")
        return
    
    if not os.path.exists(field_file):
        print(f"警告: 光场文件 {field_file} 不存在")
        print("请确保光场数据文件存在")
        return
    
    # 可视化选项
    print("\n可视化选项:")
    print("1. XY平面 (低分辨率)")
    print("2. XY平面 (中分辨率)")
    print("3. XY平面 (高分辨率)")
    print("4. 多平面视图")
    print("5. 仅轨迹 (无光场背景)")
    
    try:
        choice = input("\n请选择 (1-5, 默认1): ").strip()
        if not choice:
            choice = '1'
        
        if choice == '1':
            visualize_lp71_trajectory(trajectory_file, field_file, 'low', 'xy')
        elif choice == '2':
            visualize_lp71_trajectory(trajectory_file, field_file, 'medium', 'xy')
        elif choice == '3':
            visualize_lp71_trajectory(trajectory_file, field_file, 'high', 'xy')
        elif choice == '4':
            visualize_multiple_planes(trajectory_file, field_file, 'low')
        elif choice == '5':
            visualizer = TrajectoryVisualizer(trajectory_file)
            visualizer.plot_2d_trajectory('xy')
        else:
            print("无效选择，使用默认选项")
            visualize_lp71_trajectory(trajectory_file, field_file, 'low', 'xy')
            
    except KeyboardInterrupt:
        print("\n用户取消操作")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
