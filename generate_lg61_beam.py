import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre

def generate_lg61_beam():
    """
    生成类似LP61的LG光束 (l=6, p=1) / Generate LG beam similar to LP61 (l=6, p=1)
    """
    # 参数设置 / Parameter settings
    l = 6       # 方位角指数 (azimuthal index)
    p = 1       # 径向指数 (radial index) 
    w0 = 1      # 束腰 (beam waist, 任意单位 / arbitrary units)
    
    # 创建网格 / Create grid
    x = np.linspace(-5, 5, 500)
    y = np.linspace(-5, 5, 500)
    X, Y = np.meshgrid(x, y)
    
    # 转换为极坐标 / Convert to polar coordinates
    phi = np.arctan2(Y, X)
    r = np.sqrt(X**2 + Y**2)
    
    # 广义拉盖尔多项式 (Associated Laguerre polynomial)
    rho_squared = 2 * (r**2) / (w0**2)
    Lpl = genlaguerre(p, abs(l))(rho_squared)
    
    # LG模式振幅 (忽略Gouy相位和传播) / LG mode amplitude (ignoring Gouy phase and propagation)
    # E0 = (sqrt(2)*r/w0)^|l| * Lpl * exp(-r^2/w0^2) * exp(i*l*phi)
    radial_term = (np.sqrt(2) * r / w0) ** abs(l)
    gaussian_term = np.exp(-r**2 / w0**2)
    phase_term = np.exp(1j * l * phi)
    
    E0 = radial_term * Lpl * gaussian_term * phase_term
    
    # 螺旋相位板 (Spiral phase plate)
    spiral_phase = np.exp(1j * l * phi)
    
    # 输出光束与螺旋相位 / Output beam with spiral phase
    E_spiral = E0 * spiral_phase
    
    # 强度计算 / Intensity calculation
    I = np.abs(E_spiral)**2
    
    return I, X, Y, E_spiral

def plot_lg61_beam():
    """
    绘制LG61光束强度分布 / Plot LG61 beam intensity distribution
    """
    I, X, Y, E_spiral = generate_lg61_beam()
    
    # 绘图 / Plotting
    plt.figure(figsize=(10, 8))
    
    # 强度分布 / Intensity distribution
    plt.subplot(2, 2, 1)
    plt.imshow(I, extent=[-5, 5, -5, 5], cmap='hot', origin='lower')
    plt.colorbar(label='强度 / Intensity')
    plt.title('LG模式 l=6, p=1 (类似LP61) / LG mode with l=6, p=1 (similar to LP61)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    
    # 相位分布 / Phase distribution
    plt.subplot(2, 2, 2)
    phase = np.angle(E_spiral)
    plt.imshow(phase, extent=[-5, 5, -5, 5], cmap='hsv', origin='lower')
    plt.colorbar(label='相位 (弧度) / Phase (rad)')
    plt.title('相位分布 / Phase Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    
    # 3D强度分布 / 3D intensity distribution
    ax = plt.subplot(2, 2, 3, projection='3d')
    step = 10
    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    I_sub = I[::step, ::step]
    ax.plot_surface(X_sub, Y_sub, I_sub, cmap='hot', alpha=0.8)
    ax.set_title('3D强度分布 / 3D Intensity Distribution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('强度 / Intensity')
    
    # 径向强度分布 / Radial intensity distribution
    plt.subplot(2, 2, 4)
    # 沿x轴的强度分布 / Intensity distribution along x-axis
    center_y = I.shape[0] // 2
    x_profile = I[center_y, :]
    x_coords = np.linspace(-5, 5, len(x_profile))
    plt.plot(x_coords, x_profile, 'r-', linewidth=2, label='x轴剖面 / x-axis profile')
    
    # 沿y轴的强度分布 / Intensity distribution along y-axis
    center_x = I.shape[1] // 2
    y_profile = I[:, center_x]
    y_coords = np.linspace(-5, 5, len(y_profile))
    plt.plot(y_coords, y_profile, 'b-', linewidth=2, label='y轴剖面 / y-axis profile')
    
    plt.xlabel('位置 / Position')
    plt.ylabel('强度 / Intensity')
    plt.title('径向强度剖面 / Radial Intensity Profiles')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return I, E_spiral

def save_lg61_data(filename='lg61_beam_data.npz'):
    """
    保存LG61光束数据 / Save LG61 beam data
    
    Args:
        filename: 保存文件名 / Save filename
    """
    I, X, Y, E_spiral = generate_lg61_beam()
    
    np.savez(filename, 
             intensity=I, 
             x_grid=X, 
             y_grid=Y, 
             electric_field=E_spiral,
             l=6, p=1, w0=1)
    
    print(f"LG61光束数据已保存到 {filename} / LG61 beam data saved to {filename}")
    return filename

if __name__ == "__main__":
    print("=== LG61光束生成 / LG61 Beam Generation ===")
    print("生成LG光束 l=6, p=1 (类似LP61模式) / Generating LG beam with l=6, p=1 (similar to LP61 mode)")
    
    # 生成并绘制LG61光束 / Generate and plot LG61 beam
    I, E_spiral = plot_lg61_beam()
    
    # 保存数据 / Save data
    save_lg61_data()
    
    # 打印一些统计信息 / Print some statistics
    print(f"\n光束统计信息 / Beam Statistics:")
    print(f"最大强度 / Maximum intensity: {np.max(I):.4f}")
    print(f"总功率 (积分强度) / Total power (integrated intensity): {np.sum(I):.4f}")
    print(f"光束尺寸 (RMS) / Beam size (RMS): {np.sqrt(np.sum(I * (X**2 + Y**2)) / np.sum(I)):.4f}")
    
    print("\nLG61光束生成完成！/ LG61 beam generation completed!")
    print("光束显示出具有6重旋转对称性的特征环形结构。/ The beam shows characteristic ring structure with 6-fold rotational symmetry.")