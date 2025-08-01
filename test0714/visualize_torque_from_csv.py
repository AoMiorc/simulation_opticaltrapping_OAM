import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.constants import c

# 导入trap类
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simulation'))
from trap import OpticalTrap

def load_csv_data(intensity_file, phase_file):
    """
    加载CSV格式的强度和相位数据
    """
    try:
        # 读取强度数据
        intensity_data = pd.read_csv(intensity_file, header=None).values
        
        # 读取相位数据
        phase_data = pd.read_csv(phase_file, header=None).values
        
        print(f"Loaded intensity data shape: {intensity_data.shape}")
        print(f"Loaded phase data shape: {phase_data.shape}")
        
        return intensity_data, phase_data
        
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return None, None

def create_trap_from_csv_data(intensity_data, phase_data, x_range, y_range):
    """
    从CSV数据创建OpticalTrap对象
    """
    # 创建3D网格（添加z维度）
    z_range = np.array([0.0])  # 单个z平面
    
    # 创建trap对象
    trap = OpticalTrap(
        kappa=[1e-6, 1e-6, 1e-6],  # 阱刚度
        center=[0.0, 0.0, 0.0],    # 中心位置
        wavelength=1064e-9,        # 波长
        laser_power=0.1,           # 激光功率
        w0=2e-6,                   # 束腰半径
        l=13                       # 主要的角动量量子数（外环）
    )
    
    # 定义场函数
    def field_function(X, Y, Z):
        # 将3D坐标映射到2D CSV数据
        ny, nx = intensity_data.shape
        x_indices = np.clip(np.round((X - x_range[0]) / (x_range[-1] - x_range[0]) * (nx - 1)).astype(int), 0, nx - 1)
        y_indices = np.clip(np.round((Y - y_range[0]) / (y_range[-1] - y_range[0]) * (ny - 1)).astype(int), 0, ny - 1)
        
        # 返回强度数据
        return intensity_data[y_indices, x_indices]
    
    def phase_function(X, Y, Z):
        # 将3D坐标映射到2D CSV数据
        ny, nx = phase_data.shape
        x_indices = np.clip(np.round((X - x_range[0]) / (x_range[-1] - x_range[0]) * (nx - 1)).astype(int), 0, nx - 1)
        y_indices = np.clip(np.round((Y - y_range[0]) / (y_range[-1] - y_range[0]) * (ny - 1)).astype(int), 0, ny - 1)
        
        # 返回相位数据
        return phase_data[y_indices, x_indices]
    
    # 设置场
    trap.set_field(x_range, y_range, z_range, field_function, phase_function)
    
    return trap

def calculate_torque_field_using_trap(trap, x_range, y_range):
    """
    使用trap类的方法计算整个场的力矩分布
    """
    ny, nx = len(y_range), len(x_range)
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    
    # 初始化力矩数组
    torque_x = np.zeros((nx, ny))
    torque_y = np.zeros((nx, ny))
    torque_z = np.zeros((nx, ny))
    local_l_field = np.zeros((nx, ny))
    
    # 计算每个点的力矩
    for i in range(nx):
        for j in range(ny):
            position = [X[i, j], Y[i, j], 0.0]  # z=0平面
            
            # 使用trap类的力矩计算方法
            torque = trap.calculate_torque_at_position(position)
            torque_x[i, j] = torque[0]
            torque_y[i, j] = torque[1]
            torque_z[i, j] = torque[2]
            
            # 获取局部l值
            local_l = trap.calculate_local_l_from_phase_gradient(position)
            local_l_field[i, j] = local_l
    
    return X, Y, torque_x, torque_y, torque_z, local_l_field

def calculate_local_l_using_trap_method(phase_data, x_range, y_range, center=[0.0, 0.0, 0.0]):
    """
    使用trap类的方法计算局部l值
    """
    ny, nx = phase_data.shape
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    local_l_field = np.zeros((nx, ny))
    
    # 计算网格间距
    dx = x_range[1] - x_range[0] if len(x_range) > 1 else 1e-6
    dy = y_range[1] - y_range[0] if len(y_range) > 1 else 1e-6
    
    for i in range(1, nx-1):  # 避免边界
        for j in range(1, ny-1):
            # 当前位置
            position = [X[i, j], Y[i, j]]
            
            # 计算相位梯度（使用trap类的方法逻辑）
            phase_x_minus = phase_data[j, i-1]  # 注意索引顺序
            phase_x_plus = phase_data[j, i+1]
            phase_y_minus = phase_data[j-1, i]
            phase_y_plus = phase_data[j+1, i]
            
            # 处理相位跳变
            phase_diff_x = phase_x_plus - phase_x_minus
            phase_diff_y = phase_y_plus - phase_y_minus
            
            # 将相位差调整到 [-π, π] 范围内
            phase_diff_x = np.arctan2(np.sin(phase_diff_x), np.cos(phase_diff_x))
            phase_diff_y = np.arctan2(np.sin(phase_diff_y), np.cos(phase_diff_y))
            
            # 计算梯度
            dphase_dx = phase_diff_x / (2 * dx)
            dphase_dy = phase_diff_y / (2 * dy)
            
            # 计算到中心的距离和角度
            r_vec = np.array(position) - np.array(center[:2])
            r = np.sqrt(r_vec[0]**2 + r_vec[1]**2)
            
            if r > 1e-10:  # 避免除零
                # 计算角向梯度
                cos_theta = r_vec[0] / r
                sin_theta = r_vec[1] / r
                
                angular_gradient = sin_theta * dphase_dx - cos_theta * dphase_dy
                
                # 局部有效l值
                local_l = r * angular_gradient
                local_l_field[i, j] = local_l
            else:
                local_l_field[i, j] = 0  # 中心位置
    
    return local_l_field

def calculate_simplified_torque_from_local_l(intensity_data, local_l_field, x_range, y_range):
    """
    基于局部l值计算简化的力矩
    """
    ny, nx = intensity_data.shape
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    
    # 计算径向距离
    R = np.sqrt(X**2 + Y**2)
    
    # 计算散射截面和其他物理参数
    particle_radius = 1e-6
    wavelength = 1064e-9
    k = 2 * np.pi / wavelength
    sigma_sca = np.pi * particle_radius**2 * 0.1
    
    # 简化的力矩计算：τ_z = σ_sca * I * R * l_local / c
    # 转置local_l_field以匹配强度数据的形状
    local_l_transposed = local_l_field.T
    
    # 确保形状匹配
    if local_l_transposed.shape != intensity_data.shape:
        # 如果形状不匹配，进行插值
        from scipy.interpolate import griddata
        
        # 创建原始网格点
        x_orig = np.linspace(x_range[0], x_range[-1], local_l_field.shape[0])
        y_orig = np.linspace(y_range[0], y_range[-1], local_l_field.shape[1])
        X_orig, Y_orig = np.meshgrid(x_orig, y_orig, indexing='ij')
        
        # 创建目标网格点
        x_target = np.linspace(x_range[0], x_range[-1], intensity_data.shape[1])
        y_target = np.linspace(y_range[0], y_range[-1], intensity_data.shape[0])
        X_target, Y_target = np.meshgrid(x_target, y_target, indexing='xy')
        
        # 插值
        points = np.column_stack([X_orig.ravel(), Y_orig.ravel()])
        values = local_l_field.ravel()
        local_l_interp = griddata(points, values, (X_target, Y_target), method='linear', fill_value=0)
        
        local_l_for_calc = local_l_interp
    else:
        local_l_for_calc = local_l_transposed
    
    # 计算径向距离（匹配强度数据的网格）
    x_intensity = np.linspace(x_range[0], x_range[-1], intensity_data.shape[1])
    y_intensity = np.linspace(y_range[0], y_range[-1], intensity_data.shape[0])
    X_intensity, Y_intensity = np.meshgrid(x_intensity, y_intensity, indexing='xy')
    R_intensity = np.sqrt(X_intensity**2 + Y_intensity**2)
    
    # 计算力矩Z分量
    torque_z = sigma_sca * intensity_data * R_intensity * local_l_for_calc / c
    
    # 添加放大因子
    amplification_factor = 4.5e-13
    torque_z *= amplification_factor
    
    # X和Y分量设为零
    torque_x = np.zeros_like(torque_z)
    torque_y = np.zeros_like(torque_z)
    
    return X_intensity, Y_intensity, torque_x, torque_y, torque_z

def visualize_trap_based_torque(intensity_file, phase_file, save_plots=True):
    """
    使用trap类方法可视化力矩（简化版）
    """
    print("Starting simplified trap-based torque visualization...")
    
    # 加载数据
    intensity_data, phase_data = load_csv_data(intensity_file, phase_file)
    
    if intensity_data is None or phase_data is None:
        print("Failed to load CSV data")
        return
    
    # 设置坐标范围
    x_range = np.linspace(-6e-6, 6e-6, intensity_data.shape[1])
    y_range = np.linspace(-6e-6, 6e-6, intensity_data.shape[0])
    
    # 计算局部l值（使用trap类的方法）
    print("Calculating local l values using trap method...")
    local_l_field = calculate_local_l_using_trap_method(phase_data, x_range, y_range)
    
    # 计算简化的力矩
    print("Calculating simplified torque...")
    X, Y, torque_x, torque_y, torque_z = calculate_simplified_torque_from_local_l(
        intensity_data, local_l_field, x_range, y_range
    )
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Simplified Trap-Based Torque Analysis (LP71 Mode)', fontsize=16)
    
    # 转换为微米显示
    X_um = X * 1e6
    Y_um = Y * 1e6
    
    # 1. 强度分布
    im1 = axes[0, 0].imshow(intensity_data, extent=[-6, 6, -6, 6], 
                           origin='lower', cmap='hot', alpha=0.7)
    axes[0, 0].set_title('Intensity Distribution')
    axes[0, 0].set_xlabel('x (μm)')
    axes[0, 0].set_ylabel('y (μm)')
    plt.colorbar(im1, ax=axes[0, 0], label='Intensity')
    
    # 2. 相位分布
    im2 = axes[0, 1].imshow(phase_data, extent=[-6, 6, -6, 6], 
                           origin='lower', cmap='hsv', alpha=0.7)
    axes[0, 1].set_title('Phase Distribution')
    axes[0, 1].set_xlabel('x (μm)')
    axes[0, 1].set_ylabel('y (μm)')
    plt.colorbar(im2, ax=axes[0, 1], label='Phase (rad)')
    
    # 3. 局部l值分布
    # 为了显示，需要转置local_l_field
    local_l_display = local_l_field.T
    im3 = axes[1, 0].imshow(local_l_display, extent=[-6, 6, -6, 6], 
                           origin='lower', cmap='RdBu_r', vmin=-15, vmax=15)
    axes[1, 0].set_title('Local Angular Momentum Quantum Number (l)')
    axes[1, 0].set_xlabel('x (μm)')
    axes[1, 0].set_ylabel('y (μm)')
    plt.colorbar(im3, ax=axes[1, 0], label='Local l')
    
    # 4. 扭矩方向图
    # 创建基于力矩Z分量符号的方向图
    direction_map = np.zeros_like(torque_z)
    
    # 设置阈值
    intensity_threshold = 0.1 * np.max(intensity_data)
    torque_threshold = 0.05 * np.max(np.abs(torque_z))
    
    # 只在强度和力矩都足够大的地方显示方向
    mask = (intensity_data > intensity_threshold) & (np.abs(torque_z) > torque_threshold)
    direction_map[mask & (torque_z > 0)] = 1   # 逆时针
    direction_map[mask & (torque_z < 0)] = -1  # 顺时针
    
    # 绘制方向图
    from matplotlib.colors import ListedColormap
    colors = ['blue', 'white', 'red']  # 顺时针=蓝色，无扭矩=白色，逆时针=红色
    cmap_binary = ListedColormap(colors)
    
    im4 = axes[1, 1].imshow(direction_map, extent=[-6, 6, -6, 6], 
                           origin='lower', cmap=cmap_binary, 
                           vmin=-1, vmax=1, alpha=0.8)
    
    axes[1, 1].set_title('Torque Direction (Based on Local l)')
    axes[1, 1].set_xlabel('x (μm)')
    axes[1, 1].set_ylabel('y (μm)')
    axes[1, 1].set_xlim([-6, 6])
    axes[1, 1].set_ylim([-6, 6])
    
    # 添加图例
    axes[1, 1].text(-5.5, 5.5, 'Red: Counter-clockwise (l>0)\nBlue: Clockwise (l<0)', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                    fontsize=9, verticalalignment='top')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('simplified_trap_torque_analysis.png', dpi=300, bbox_inches='tight')
        print("Simplified trap torque analysis saved as 'simplified_trap_torque_analysis.png'")
    
    # 打印统计信息
    print("\n=== Simplified Trap-Based Torque Statistics ===")
    print(f"Local l range: [{np.min(local_l_field):.2f}, {np.max(local_l_field):.2f}]")
    torque_magnitude = np.sqrt(torque_x**2 + torque_y**2 + torque_z**2)
    print(f"Maximum torque magnitude: {np.max(torque_magnitude):.2e} N⋅m")
    print(f"Torque Z range: [{np.min(torque_z):.2e}, {np.max(torque_z):.2e}] N⋅m")
    
    # 分析内外环的l值分布
    r_field = np.sqrt(X**2 + Y**2)
    inner_mask = r_field < 1.5e-6
    outer_mask = r_field > 2.5e-6
    
    # 使用local_l_display进行分析
    x_analysis = np.linspace(-6e-6, 6e-6, local_l_display.shape[1])
    y_analysis = np.linspace(-6e-6, 6e-6, local_l_display.shape[0])
    X_analysis, Y_analysis = np.meshgrid(x_analysis, y_analysis, indexing='xy')
    r_analysis = np.sqrt(X_analysis**2 + Y_analysis**2)
    
    inner_mask_analysis = r_analysis < 1.5e-6
    outer_mask_analysis = r_analysis > 2.5e-6
    
    if np.any(inner_mask_analysis):
        inner_l_avg = np.mean(local_l_display[inner_mask_analysis])
        print(f"Inner ring average l: {inner_l_avg:.2f}")
    
    if np.any(outer_mask_analysis):
        outer_l_avg = np.mean(local_l_display[outer_mask_analysis])
        print(f"Outer ring average l: {outer_l_avg:.2f}")
    
    plt.show()
    
    return X, Y, torque_x, torque_y, torque_z, local_l_field

if __name__ == "__main__":
    # 文件路径
    intensity_file = os.path.join(os.path.dirname(__file__), "final_intensity_LP71_m6_10cm.csv")
    phase_file = os.path.join(os.path.dirname(__file__), "final_phase_LP71_m6_10cm.csv")
    
    # 检查文件是否存在
    if not os.path.exists(intensity_file):
        print(f"Intensity file not found: {intensity_file}")
        exit(1)
    if not os.path.exists(phase_file):
        print(f"Phase file not found: {phase_file}")
        exit(1)
    
    # 运行简化的基于trap类的力矩分析
    print("Running simplified trap-based torque analysis...")
    X, Y, torque_x, torque_y, torque_z, local_l_field = visualize_trap_based_torque(
        intensity_file, phase_file
    )