import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import os

def load_lp31_csv_data(filename):
    """
    加载LP31 CSV光场数据
    Load LP31 CSV optical field data
    """
    try:
        intensity_data = np.loadtxt(filename, delimiter=',')
        print(f"成功加载LP31光场数据，数据形状: {intensity_data.shape}")
        return intensity_data
    except Exception as e:
        print(f"加载CSV文件失败: {e}")
        return None

def setup_lp31_field_interpolator(intensity_data, x_range, y_range):
    """
    设置LP31光场插值器
    Setup LP31 optical field interpolator
    """
    if len(intensity_data.shape) == 2:
        ny, nx = intensity_data.shape
        
        # 创建对应的坐标
        x_csv = np.linspace(x_range[0], x_range[-1], nx)
        y_csv = np.linspace(y_range[0], y_range[-1], ny)
        
        # 创建插值函数
        interpolator = RegularGridInterpolator(
            (y_csv, x_csv),  # 注意顺序：y在前，x在后
            intensity_data, 
            method='linear', 
            bounds_error=False, 
            fill_value=0.0
        )
        
        return interpolator, x_csv, y_csv
    else:
        print(f"不支持的数据格式: {intensity_data.shape}")
        return None, None, None

def visualize_lp31_field(filename="final_intensity_LP31_2cm.csv", 
                        field_range=5e-6, resolution=200):
    """
    可视化LP31光场数据
    Visualize LP31 optical field data
    
    参数:
    filename: CSV文件名
    field_range: 显示范围 (±field_range)
    resolution: 网格分辨率
    """
    
    print("=== LP31光场可视化 / LP31 Optical Field Visualization ===")
    
    # 1. 加载数据
    intensity_data = load_lp31_csv_data(filename)
    if intensity_data is None:
        return
    
    # 2. 设置坐标范围
    x_range = np.linspace(-field_range, field_range, resolution)
    y_range = np.linspace(-field_range, field_range, resolution)
    
    # 3. 设置插值器
    interpolator, x_csv, y_csv = setup_lp31_field_interpolator(
        intensity_data, x_range, y_range
    )
    
    if interpolator is None:
        return
    
    # 4. 创建网格
    X, Y = np.meshgrid(x_range, y_range)
    
    # 5. 插值得到完整的光场分布
    points = np.column_stack([Y.ravel(), X.ravel()])
    Z = interpolator(points).reshape(X.shape)
    
    # 6. 数据统计
    print(f"\n=== LP31光场数据统计 / LP31 Field Data Statistics ===")
    print(f"原始数据形状: {intensity_data.shape}")
    print(f"插值后数据形状: {Z.shape}")
    print(f"最大强度: {np.max(Z):.2e}")
    print(f"最小强度: {np.min(Z):.2e}")
    print(f"平均强度: {np.mean(Z):.2e}")
    print(f"强度标准差: {np.std(Z):.2e}")
    
    # 7. 创建多子图可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('LP31光场强度分布可视化 / LP31 Optical Field Intensity Distribution', 
                 fontsize=16, fontweight='bold')
    
    # 7.1 线性刻度强度图
    im1 = axes[0, 0].contourf(X*1e6, Y*1e6, Z, levels=50, cmap='hot')
    axes[0, 0].set_title('LP31强度分布 (线性刻度)\nLP31 Intensity (Linear Scale)')
    axes[0, 0].set_xlabel('X (μm)')
    axes[0, 0].set_ylabel('Y (μm)')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0], label='强度 / Intensity')
    
    # 7.2 对数刻度强度图
    Z_log = np.log10(np.maximum(Z, np.max(Z)*1e-6))  # 避免log(0)
    im2 = axes[0, 1].contourf(X*1e6, Y*1e6, Z_log, levels=50, cmap='hot')
    axes[0, 1].set_title('LP31强度分布 (对数刻度)\nLP31 Intensity (Log Scale)')
    axes[0, 1].set_xlabel('X (μm)')
    axes[0, 1].set_ylabel('Y (μm)')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 1], label='log₁₀(强度) / log₁₀(Intensity)')
    
    # 7.3 等高线图
    contour_levels = np.linspace(np.min(Z), np.max(Z), 20)
    cs = axes[0, 2].contour(X*1e6, Y*1e6, Z, levels=contour_levels, colors='black', linewidths=0.8)
    axes[0, 2].clabel(cs, inline=True, fontsize=8, fmt='%.1e')
    axes[0, 2].set_title('LP31等高线图\nLP31 Contour Lines')
    axes[0, 2].set_xlabel('X (μm)')
    axes[0, 2].set_ylabel('Y (μm)')
    axes[0, 2].set_aspect('equal')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 7.4 中心截面图 (X方向)
    center_y_idx = len(y_range) // 2
    x_profile = Z[center_y_idx, :]
    axes[1, 0].plot(x_range*1e6, x_profile, 'b-', linewidth=2, label='Y=0截面')
    axes[1, 0].set_title('LP31中心截面 (X方向)\nLP31 Central Cross-section (X-direction)')
    axes[1, 0].set_xlabel('X (μm)')
    axes[1, 0].set_ylabel('强度 / Intensity')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 7.5 中心截面图 (Y方向)
    center_x_idx = len(x_range) // 2
    y_profile = Z[:, center_x_idx]
    axes[1, 1].plot(y_range*1e6, y_profile, 'r-', linewidth=2, label='X=0截面')
    axes[1, 1].set_title('LP31中心截面 (Y方向)\nLP31 Central Cross-section (Y-direction)')
    axes[1, 1].set_xlabel('Y (μm)')
    axes[1, 1].set_ylabel('强度 / Intensity')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # 7.6 3D表面图
    ax_3d = fig.add_subplot(2, 3, 6, projection='3d')
    # 降采样以提高性能
    step = max(1, resolution // 50)
    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    Z_sub = Z[::step, ::step]
    
    surf = ax_3d.plot_surface(X_sub*1e6, Y_sub*1e6, Z_sub, 
                             cmap='hot', alpha=0.8, 
                             linewidth=0, antialiased=True)
    ax_3d.set_title('LP31 3D表面图\nLP31 3D Surface')
    ax_3d.set_xlabel('X (μm)')
    ax_3d.set_ylabel('Y (μm)')
    ax_3d.set_zlabel('强度 / Intensity')
    
    plt.tight_layout()
    
    # 8. 保存图像
    output_filename = f"lp31_field_visualization_{field_range*1e6:.0f}um.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n可视化图像已保存为: {output_filename}")
    
    plt.show()
    
    # 9. 分析LP31光场特征
    print(f"\n=== LP31光场特征分析 / LP31 Field Characteristics Analysis ===")
    
    # 找到强度峰值位置
    max_idx = np.unravel_index(np.argmax(Z), Z.shape)
    max_x = x_range[max_idx[1]]
    max_y = y_range[max_idx[0]]
    print(f"最大强度位置: ({max_x*1e6:.2f}, {max_y*1e6:.2f}) μm")
    
    # 计算径向分布
    center_x_idx = len(x_range) // 2
    center_y_idx = len(y_range) // 2
    
    # 从中心开始的径向距离
    r_max = min(field_range, field_range)
    r_points = np.linspace(0, r_max, 100)
    radial_intensity = []
    
    for r in r_points:
        if r == 0:
            radial_intensity.append(Z[center_y_idx, center_x_idx])
        else:
            # 在半径r的圆周上采样
            angles = np.linspace(0, 2*np.pi, 36)
            intensities = []
            for angle in angles:
                x_sample = r * np.cos(angle)
                y_sample = r * np.sin(angle)
                
                # 插值得到该点的强度
                if abs(x_sample) <= field_range and abs(y_sample) <= field_range:
                    intensity = interpolator(np.array([[y_sample, x_sample]]))[0]
                    intensities.append(intensity)
            
            if intensities:
                radial_intensity.append(np.mean(intensities))
            else:
                radial_intensity.append(0)
    
    # 绘制径向分布
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(r_points)*1e6, radial_intensity, 'b-', linewidth=2)
    plt.title('LP31径向强度分布 / LP31 Radial Intensity Distribution')
    plt.xlabel('径向距离 / Radial Distance (μm)')
    plt.ylabel('平均强度 / Average Intensity')
    plt.grid(True, alpha=0.3)
    
    # 保存径向分布图
    radial_filename = f"lp31_radial_distribution_{field_range*1e6:.0f}um.png"
    plt.savefig(radial_filename, dpi=300, bbox_inches='tight')
    print(f"径向分布图已保存为: {radial_filename}")
    plt.show()
    
    print(f"\n=== LP31模式特性 / LP31 Mode Characteristics ===")
    print(f"轨道角动量量子数: l = 3")
    print(f"径向量子数: p = 1")
    print(f"LP31模式具有3重对称性和环形强度分布")
    print(f"相比LP71 (l=7)，LP31的角动量较小，旋转效应相对温和")
    
    return Z, X, Y

def compare_lp31_with_theory(csv_filename="final_intensity_LP31_2cm.csv", 
                           field_range=5e-6, resolution=100):
    """
    比较CSV数据与理论LP31模式
    Compare CSV data with theoretical LP31 mode
    """
    print("\n=== LP31 CSV数据与理论模式比较 ===")
    
    # 加载CSV数据
    intensity_data = load_lp31_csv_data(csv_filename)
    if intensity_data is None:
        return
    
    # 设置坐标
    x_range = np.linspace(-field_range, field_range, resolution)
    y_range = np.linspace(-field_range, field_range, resolution)
    X, Y = np.meshgrid(x_range, y_range)
    
    # CSV数据插值
    interpolator, _, _ = setup_lp31_field_interpolator(intensity_data, x_range, y_range)
    if interpolator is None:
        return
    
    points = np.column_stack([Y.ravel(), X.ravel()])
    Z_csv = interpolator(points).reshape(X.shape)
    
    # 理论LP31模式
    def create_theoretical_lp31(x, y, w0=2e-6, l=3, p=1):
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        
        # 避免除零
        r = np.maximum(r, 1e-12)
        
        # 归一化径向坐标
        rho = np.sqrt(2) * r / w0
        
        # LP31模式的径向部分 (l=3, p=1)
        laguerre_part = 4 - rho**2  # L_1^3(ρ²)
        radial_part = rho**3 * laguerre_part * np.exp(-rho**2 / 2)
        
        # 角向部分
        angular_part = np.exp(1j * l * phi)
        
        # 高斯包络
        gaussian_envelope = np.exp(-r**2 / w0**2)
        
        # 完整场振幅
        amplitude = gaussian_envelope * radial_part * angular_part
        
        return np.abs(amplitude)**2
    
    Z_theory = create_theoretical_lp31(X, Y)
    
    # 归一化以便比较
    Z_csv_norm = Z_csv / np.max(Z_csv)
    Z_theory_norm = Z_theory / np.max(Z_theory)
    
    # 比较可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('LP31 CSV数据与理论模式比较', fontsize=14, fontweight='bold')
    
    # CSV数据
    im1 = axes[0, 0].contourf(X*1e6, Y*1e6, Z_csv_norm, levels=50, cmap='hot')
    axes[0, 0].set_title('CSV数据 (归一化)')
    axes[0, 0].set_xlabel('X (μm)')
    axes[0, 0].set_ylabel('Y (μm)')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 理论模式
    im2 = axes[0, 1].contourf(X*1e6, Y*1e6, Z_theory_norm, levels=50, cmap='hot')
    axes[0, 1].set_title('理论LP31模式 (归一化)')
    axes[0, 1].set_xlabel('X (μm)')
    axes[0, 1].set_ylabel('Y (μm)')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 差异图
    diff = Z_csv_norm - Z_theory_norm
    im3 = axes[1, 0].contourf(X*1e6, Y*1e6, diff, levels=50, cmap='RdBu_r')
    axes[1, 0].set_title('差异 (CSV - 理论)')
    axes[1, 0].set_xlabel('X (μm)')
    axes[1, 0].set_ylabel('Y (μm)')
    axes[1, 0].set_aspect('equal')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 中心截面比较
    center_idx = resolution // 2
    axes[1, 1].plot(x_range*1e6, Z_csv_norm[center_idx, :], 'b-', linewidth=2, label='CSV数据')
    axes[1, 1].plot(x_range*1e6, Z_theory_norm[center_idx, :], 'r--', linewidth=2, label='理论模式')
    axes[1, 1].set_title('中心截面比较 (Y=0)')
    axes[1, 1].set_xlabel('X (μm)')
    axes[1, 1].set_ylabel('归一化强度')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存比较图
    comparison_filename = f"lp31_csv_vs_theory_comparison_{field_range*1e6:.0f}um.png"
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
    print(f"比较图已保存为: {comparison_filename}")
    plt.show()
    
    # 计算相似度指标
    correlation = np.corrcoef(Z_csv_norm.ravel(), Z_theory_norm.ravel())[0, 1]
    mse = np.mean((Z_csv_norm - Z_theory_norm)**2)
    
    print(f"\n=== 相似度分析 ===")
    print(f"相关系数: {correlation:.4f}")
    print(f"均方误差: {mse:.6f}")
    print(f"最大差异: {np.max(np.abs(diff)):.4f}")
    
    if correlation > 0.8:
        print("CSV数据与理论模式高度相似")
    elif correlation > 0.6:
        print("CSV数据与理论模式中等相似")
    else:
        print("CSV数据与理论模式差异较大")

# 主程序
if __name__ == "__main__":
    print("LP31光场可视化工具 / LP31 Optical Field Visualization Tool")
    print("=" * 60)
    
    # 检查文件是否存在
    csv_file = "final_intensity_LP31_2cm.csv"
    if not os.path.exists(csv_file):
        print(f"错误: 找不到文件 {csv_file}")
        print("请确保CSV文件在当前目录中")
    else:
        print(f"找到LP31光场数据文件: {csv_file}")
        
        # 基本可视化
        print("\n1. 基本LP31光场可视化...")
        Z, X, Y = visualize_lp31_field(csv_file, field_range=5e-6, resolution=200)
        
        # 与理论模式比较
        print("\n2. 与理论LP31模式比较...")
        compare_lp31_with_theory(csv_file, field_range=5e-6, resolution=100)
        
        print("\n=== 使用说明 ===")
        print("1. 此脚本专门用于可视化LP31光场的CSV数据")
        print("2. 生成多种可视化图表：线性/对数刻度、等高线、截面图、3D表面")
        print("3. 提供径向分布分析和与理论模式的比较")
        print("4. 所有图像自动保存为PNG格式")
        print("5. 可以通过修改field_range和resolution参数调整显示范围和精度")
        
        print("\nLP31光场可视化完成！")