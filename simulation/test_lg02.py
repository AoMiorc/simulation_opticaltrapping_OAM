from simulation import Environment, Particle, OpticalTrap, SimulationBox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 创建环境（液体）
liquid_env = Environment(
    medium='liquid',        # 液体环境
    T=298.0,               # 温度298K
    eta=0.00001              # 水的粘度
)

# 创建粒子
particle_mass = 1000 * (4/3) * np.pi * (5e-7)**3
particle = Particle(mass=particle_mass, radius=5e-7)

# 创建光阱
optical_trap = OpticalTrap(
    kappa=[1e-5, 1e-5, 0.5e-5],
    wavelength=1064e-9,
    laser_power=0.05,    
    w0=2e-6,
    l=2                  # 设置为LG02模式
)

# 定义LG02光场分布函数
def lg02_field(x, y, z):
    # 计算r和θ
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    w0 = 2e-6  # 束腰半径
    
    # LG02模式（p=0, l=2）
    amplitude = (r**2/w0**2) * np.exp(-r**2/w0**2)  # r^2 for l=2
    phase = np.exp(6j * theta)  # 2j for l=2
    lg_field = (np.abs(amplitude * phase))**2
    
    return lg_field

# 设置光场
x_grid = np.linspace(-3e-6, 3e-6, 150)
y_grid = np.linspace(-3e-6, 3e-6, 150)
z_grid = np.linspace(-2e-6, 2e-6, 20)
optical_trap.set_field(x_grid, y_grid, z_grid, lg02_field)

# 绘制二维光场分布
def plot_2d_field(x_grid, y_grid, field_func):
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)  # z=0平面
    intensity = field_func(X, Y, Z)
    
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X*1e6, Y*1e6, intensity, cmap='hot')
    plt.colorbar(label='Normalized Intensity')
    plt.xlabel('X (μm)')
    plt.ylabel('Y (μm)')
    plt.title('LG02 Mode Intensity Distribution (z=0)')
    plt.axis('equal')
    plt.show()

# 绘制二维光场分布
plot_2d_field(x_grid, y_grid, lg02_field)

# 绘制三维光场分布
# optical_trap.plot_3d_field(num_points=50)

# 创建模拟系统
sim_box = SimulationBox(particle, optical_trap, liquid_env)

# 设置初始位置和速度
sim_box.particle.position = np.array([1.5e-6, 1.5e-6, 0])
sim_box.particle.velocity = np.array([-0.01, -0.01, 0])

# 运行模拟
# 运行模拟
trajectory_data = sim_box.simulate(duration=1e-3, timestep=1e-6)  # 改回原来的值

# 创建实时动画
# 创建实时动画
sim_box.animate_trajectory_3d(duration=5e-3, timestep=1e-7)  # 将 duration 从 1e-4 增加到 1e-3，使模拟时间延长10倍


# 修改二维光场可视化函数
def plot_radial_intensity(x_grid, y_grid, field_func):
    # Get beam waist radius
    w0 = 2e-6  # Same as in lg02_field function
    
    # Create radial coordinates
    r = np.linspace(0, 3e-6, 300)  # 0 to 3 microns, 300 points
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Calculate average intensity at each r
    intensity = np.zeros_like(r)
    for i, r_val in enumerate(r):
        # Take average over multiple angles at each r
        x = r_val * np.cos(theta)
        y = r_val * np.sin(theta)
        z = np.zeros_like(x)
        intensity[i] = np.mean(field_func(x, y, z))
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot intensity distribution
    plt.plot(r*1e6, intensity, 'b-', linewidth=2, label='Intensity Distribution')
    
    # Find maximum intensity position
    max_idx = np.argmax(intensity)
    r_max = r[max_idx]
    I_max = intensity[max_idx]
    
    # Mark important features
    plt.plot(r_max*1e6, I_max, 'ro', label=f'Max Intensity: r = {r_max*1e6:.1f} μm')
    
    # Calculate FWHM
    half_max = I_max/2
    r_half = r[intensity > half_max]
    fwhm = (r_half[-1] - r_half[0])
    
    # Add FWHM annotation
    plt.axhline(y=half_max, color='g', linestyle='--', alpha=0.5)
    plt.axvline(x=r_half[0]*1e6, color='g', linestyle='--', alpha=0.5)
    plt.axvline(x=r_half[-1]*1e6, color='g', linestyle='--', alpha=0.5)
    
    # Set labels and title
    plt.xlabel('Radial Distance r (μm)')
    plt.ylabel('Normalized Intensity I (a.u.)')
    plt.title('LG02 Mode Radial Intensity Distribution')
    
    # Add text annotation
    plt.text(0.98, 0.95, 
             f'Beam Characteristics:\n' + 
             f'Max Intensity Position: {r_max*1e6:.1f} μm\n' + 
             f'FWHM: {fwhm*1e6:.1f} μm\n' + 
             f'Beam Waist (w0): {w0*1e6:.1f} μm', 
             transform=plt.gca().transAxes,
             horizontalalignment='right',
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# 调用新的可视化函数
plot_radial_intensity(x_grid, y_grid, lg02_field)
