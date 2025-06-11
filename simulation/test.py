from simulation import Environment,Particle,OpticalTrap,SimulationBox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 创建环境（气体）
gas_env = Environment(
    medium='gas',          
    T=298.0,               
    eta=1.8e-5,            
    P_gas=101325,         
    M_gas=4.8e-26          
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
    l=1                 
)

# 定义LG光场分布函数
def lg_field(x, y, z):
    # 计算r和θ
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    w0 = 2e-6  # 束腰半径
    
    # LG01模式（p=0, l=1）
    amplitude = (r/w0) * np.exp(-r**2/w0**2)
    phase = np.exp(1j * theta)
    lg_field = (np.abs(amplitude * phase))**2
    
    return lg_field

# 设置光场
x_grid = np.linspace(-3e-6, 3e-6, 150)
y_grid = np.linspace(-3e-6, 3e-6, 150)
z_grid = np.linspace(-2e-6, 2e-6, 20)
optical_trap.set_field(x_grid, y_grid, z_grid, lg_field)


# 创建模拟系统
sim_box = SimulationBox(particle, optical_trap, gas_env)

# 设置初始位置和速度
sim_box.particle.position = np.array([1.5e-6, 1.5e-6, 0])  # 从2.5e-6改为1.5e-6
sim_box.particle.velocity = np.array([-0.01, -0.01, 0])

# 绘制光场
optical_trap.plot_3d_field(num_points=50)

# 运行模拟
trajectory_data = sim_box.simulate(duration=1e-3, timestep=1e-6)  # 改回原来的值

# 创建实时动画
sim_box.animate_trajectory(duration=1e-4, timestep=1e-7)  # 改回原来的值