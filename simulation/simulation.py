import numpy as np
from scipy.constants import Boltzmann as k_B
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation  # 添加这行导入语句

class Environment:
    """表示粒子所在的环境（液体或气体）"""
    def __init__(self, medium='liquid', T=298.0, eta=0.001, P_gas=101325.0, M_gas=4.8e-26):
        """
        初始化环境参数
        
        参数:
        medium (str): 介质类型，'liquid'或'gas'
        T (float): 环境温度 (K)
        eta (float): 粘度 (Pa·s)
        P_gas (float): 气体压力 (Pa)，仅对气体介质有效
        M_gas (float): 气体分子质量 (kg)，仅对气体介质有效
        """
        self.medium = medium
        self.T = T  # 环境温度 (K)
        self.eta = eta  # 粘度 (Pa·s)
        self.P_gas = P_gas  # 气体压力 (Pa)
        self.M_gas = M_gas  # 气体分子质量 (kg)
    
    def get_drag_coefficient(self, particle):
        """计算阻尼系数γ_q，根据介质类型使用不同的公式"""
        a = particle.radius
        
        if self.medium == 'liquid':
            # 液体环境：斯托克斯定律
            return 6 * np.pi * a * self.eta
        
        elif self.medium == 'gas':
            # 气体环境：计算克努森数和阻尼率
            # 计算平均自由程
            mean_free_path = (self.eta / self.P_gas) * np.sqrt(np.pi * k_B * self.T / (2 * self.M_gas))
            
            # 克努森数
            Kn = mean_free_path / a
            
            # 完整阻尼率公式
            term1 = 0.619 / (0.619 + Kn)
            term2 = 1 + (0.31 * Kn) / (0.785 + 1.152 * Kn + Kn**2)
            Gamma_q = (6 * np.pi * a * self.eta / particle.mass) * term1 * term2
            
            # 对于低压力情况使用近似公式
            if Kn > 10:  # Kn >> 1
                Gamma_q = 3.714 * (a**2 / particle.mass) * np.sqrt(
                    2 * np.pi * self.M_gas / (k_B * self.T)) * self.P_gas
            
            return Gamma_q * particle.mass  # 转换为阻尼系数γ_q = Γ_q * M
        
        else:
            raise ValueError(f"未知介质类型: {self.medium}")


class Particle:
    """表示被捕获的粒子"""
    def __init__(self, mass, radius, position=None):
        """初始化粒子
        
        参数:
        mass (float): 粒子质量 (kg)
        radius (float): 粒子半径 (m)
        position (np.array): 初始位置 (m)，默认为原点
        """
        self.mass = mass
        self.radius = radius
        self.position = np.array([0.0, 0.0, 0.0]) if position is None else position
        self.velocity = np.array([0.0, 0.0, 0.0])  # 初始速度
        self.acceleration = np.array([0.0, 0.0, 0.0])  # 初始加速度
        self.force = np.array([0.0, 0.0, 0.0])  # 当前受力
        
        # 添加角运动相关属性
        self.moment_of_inertia = (2/5) * mass * radius**2  # 球形粒子的转动惯量
        self.angular_velocity = np.array([0.0, 0.0, 0.0])  # 角速度
        self.angular_acceleration = np.array([0.0, 0.0, 0.0])  # 角加速度
        self.torque = np.array([0.0, 0.0, 0.0])  # 当前扭矩
        self.orientation = np.array([0.0, 0.0, 0.0])  # 初始化欧拉角姿态


# 在文件开头添加光速常数
from scipy.constants import c  # 光速
# 在文件开头添加
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class OpticalTrap:
    """表示光阱及其属性"""
    def __init__(self, kappa, center=None, wavelength=1064e-9, laser_power=0.1, w0=2e-6, l=1):
        """初始化光阱
        
        参数:
        kappa (np.array): 三个方向的阱刚度 [κ_x, κ_y, κ_z] (N/m)
        center (np.array): 阱中心位置 [x, y, z] (m)，默认为原点
        wavelength (float): 激光波长 (m)，默认1064nm
        laser_power (float): 激光功率 (W)，默认0.1W
        w0 (float): 束腰半径 (m)，默认2微米
        l (int): 轨道角动量量子数，默认1
        """
        self.kappa = np.array(kappa)  # 阱刚度
        self.center = np.array([0.0, 0.0, 0.0]) if center is None else center
        self.field = None  # 光场矩阵（将在set_field方法中设置）
        self.wavelength = wavelength
        self.laser_power = laser_power
        self.w0 = w0
        self.l = l  # 轨道角动量量子数
    
    def get_intensity_at_position(self, position):
        """获取指定位置的归一化光强"""
        if self.field is None:
            return 0.0
            
        # 找到最近的网格点
        x_idx = np.searchsorted(self.grid_x, position[0])
        y_idx = np.searchsorted(self.grid_y, position[1])
        z_idx = np.searchsorted(self.grid_z, position[2])
        
        # 确保索引在有效范围内
        x_idx = np.clip(x_idx, 0, len(self.grid_x)-1)
        y_idx = np.clip(y_idx, 0, len(self.grid_y)-1)
        z_idx = np.clip(z_idx, 0, len(self.grid_z)-1)
        
        return self.field[x_idx, y_idx, z_idx]

    def get_torque(self, particle):
        """计算粒子在当前位置受到的扭矩"""
        # 获取粒子位置处的光强
        normalized_intensity = self.get_intensity_at_position(particle.position)
        actual_intensity = (self.laser_power / (np.pi * self.w0**2)) * normalized_intensity
        
        # 计算激光角频率
        omega_laser = 2 * np.pi * c / self.wavelength
        
        # 计算总截面
        total_cross_section = np.pi * particle.radius**2  # 简化模型，使用几何截面
        
        # 计算扭矩大小
        torque_magnitude = (actual_intensity * total_cross_section / omega_laser) * self.l
        
        # 扭矩方向：垂直于位置矢量，在xy平面内
        dx = particle.position[0] - self.center[0]
        dy = particle.position[1] - self.center[1]
        r = np.sqrt(dx**2 + dy**2)
        
        if r > 0:
            # 单位切向向量（与角动量转移方向一致）
            tangential_dir = np.array([-dy/r, dx/r, 0])
            torque = torque_magnitude * tangential_dir
        else:
            torque = np.zeros(3)
        
        return torque

    def plot_3d_field(self, num_points=50):
        """绘制光阱的三维图像，包含等值面和体渲染
        
        参数:
        num_points (int): 每个维度上的采样点数，默认50
        """
        if self.field is None:
            print("错误：光场未初始化，请先调用 set_field 方法设置光场")
            return
            
        # 创建图形窗口
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建采样点
        x = np.linspace(self.grid_x[0], self.grid_x[-1], num_points)
        y = np.linspace(self.grid_y[0], self.grid_y[-1], num_points)
        z = np.linspace(self.grid_z[0], self.grid_z[-1], num_points)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 对光场进行插值
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (self.grid_x, self.grid_y, self.grid_z),
            self.field
        )
        
        # 计算插值后的光场强度
        points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        intensity = interp(points).reshape(X.shape)
        
        # 1. 添加体渲染
        # 选择一些随机点进行散点图绘制，以避免过度密集
        mask = np.random.random(size=intensity.shape) < 0.1  # 只显示10%的点
        x_plot = X[mask]
        y_plot = Y[mask]
        z_plot = Z[mask]
        i_plot = intensity[mask]
        
        # 归一化强度值
        i_norm = i_plot / np.max(i_plot)
        
        # 使用散点图实现体渲染效果
        scatter = ax.scatter(x_plot, y_plot, z_plot,
                            c=i_norm,
                            cmap='plasma',
                            alpha=0.1,
                            s=10)
        plt.colorbar(scatter, label='Normalized Intensity')
        
        # 2. 保留原有的等值面绘制
        max_intensity = np.max(intensity)
        levels = [0.3, 0.5, 0.7]  # 设置等值面的相对强度水平
        
        for level in levels:
            verts, faces, _, _ = measure.marching_cubes(intensity/max_intensity, level=level)
            # 缩放顶点坐标到实际物理尺寸
            verts[:, 0] = x[0] + (x[-1] - x[0]) * verts[:, 0] / (num_points - 1)
            verts[:, 1] = y[0] + (y[-1] - y[0]) * verts[:, 1] / (num_points - 1)
            verts[:, 2] = z[0] + (z[-1] - z[0]) * verts[:, 2] / (num_points - 1)
            # 绘制等值面
            mesh = Poly3DCollection(verts[faces], alpha=0.2)  # 降低等值面的不透明度
            mesh.set_facecolor('orange')
            ax.add_collection3d(mesh)
        
        # 绘制光阱中心
        ax.scatter([self.center[0]], [self.center[1]], [self.center[2]],
                  color='red', marker='o', s=100, label='Trap Center')
        
        # 设置标签和标题
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Optical Trap Field Distribution')
        
        # 设置坐标轴范围
        ax.set_xlim([x[0], x[-1]])
        ax.set_ylim([y[0], y[-1]])
        ax.set_zlim([z[0], z[-1]])
        
        # 添加图例
        ax.legend()
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        # 调整布局并显示
        plt.tight_layout()
        plt.show()
    
    def set_field(self, grid_x, grid_y, grid_z, field_function):
        """
        设置光场矩阵
        
        参数:
        grid_x, grid_y, grid_z (np.array): 空间网格坐标
        field_function (callable): 计算光场强度的函数 f(x, y, z)
        """
        # 创建网格
        X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
        
        # 计算整个网格上的光场强度
        self.field = field_function(X, Y, Z)
        
        # 存储网格信息
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
    
    def get_force(self, position):
        """计算光学力
        F = α∇I(r)，其中α是极化率相关的系数，I(r)是光场强度
        """
        if self.field is None:
            return np.zeros(3)
            
        # 计算极化率相关的系数（这里使用一个简化的模型）
        alpha = self.kappa[0] * 1e-12  # 使用x方向的kappa值来估计极化率系数
        
        # 使用中心差分法计算光强梯度
        dx = self.grid_x[1] - self.grid_x[0]
        dy = self.grid_y[1] - self.grid_y[0]
        dz = self.grid_z[1] - self.grid_z[0]
        
        # 在三个方向上分别计算梯度
        grad_x = np.zeros(3)
        grad_y = np.zeros(3)
        grad_z = np.zeros(3)
        
        # 找到最近的网格点
        x_idx = np.searchsorted(self.grid_x, position[0])
        y_idx = np.searchsorted(self.grid_y, position[1])
        z_idx = np.searchsorted(self.grid_z, position[2])
        
        # 确保索引在有效范围内
        x_idx = np.clip(x_idx, 1, len(self.grid_x)-2)
        y_idx = np.clip(y_idx, 1, len(self.grid_y)-2)
        z_idx = np.clip(z_idx, 1, len(self.grid_z)-2)
        
        # 计算x方向的梯度
        intensity_right = self.field[x_idx+1, y_idx, z_idx]
        intensity_left = self.field[x_idx-1, y_idx, z_idx]
        grad_x[0] = (intensity_right - intensity_left) / (2*dx)
        
        # 计算y方向的梯度
        intensity_up = self.field[x_idx, y_idx+1, z_idx]
        intensity_down = self.field[x_idx, y_idx-1, z_idx]
        grad_x[1] = (intensity_up - intensity_down) / (2*dy)
        
        # 计算z方向的梯度
        intensity_front = self.field[x_idx, y_idx, z_idx+1]
        intensity_back = self.field[x_idx, y_idx, z_idx-1]
        grad_x[2] = (intensity_front - intensity_back) / (2*dz)
        
        # 合成总梯度
        gradient = grad_x + grad_y + grad_z
        
        # 计算光学力
        force = alpha * gradient
        
        
        return force
    
    def get_field_2d(self, x, y):
        """获取二维平面上的光场分布"""
        if self.field is None:
            return None
        # 找到最接近z=0的网格点
        z_mid_index = len(self.grid_z) // 2
        # 插值得到xy平面上的场分布
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator((self.grid_x, self.grid_y), self.field[:,:,z_mid_index])
        return interp((x, y))


class SimulationBox:
    """三维模拟画布，整合粒子、光阱和环境"""
    def __init__(self, particle, optical_trap, environment):
        """初始化模拟环境
        
        参数:
        particle (Particle): 粒子实例
        optical_trap (OpticalTrap): 光阱实例
        environment (Environment): 环境实例
        """
        self.particle = particle
        self.optical_trap = optical_trap
        self.environment = environment
        self.time = 0  # 初始时间 (s)
        self.timestep = 1e-6  # 改回默认时间步长
        self.trajectory = []  # 存储粒子轨迹 [时间, 位置]
        self.velocity_history = []  # 存储速度历史
        self.force_history = []  # 存储受力历史
        self.angular_trajectory = []  # 存储角速度历史
        self.torque_history = []  # 存储力矩历史
        
        # 初始阻尼系数（三个方向相同）
        self.gamma = np.array([
            environment.get_drag_coefficient(particle),
            environment.get_drag_coefficient(particle),
            environment.get_drag_coefficient(particle)
        ])
    
    def update_damping(self):
        """更新阻尼系数（如果环境参数变化）"""
        self.gamma = np.array([
            self.environment.get_drag_coefficient(self.particle),
            self.environment.get_drag_coefficient(self.particle),
            self.environment.get_drag_coefficient(self.particle)
        ])
    
    def calculate_fluctuation_force(self):
        """计算随机涨落力（布朗力）"""
        # 计算随机力的方差
        variance = 2 * self.gamma * k_B * self.environment.T / self.timestep
        
        # 生成高斯随机力（三个方向独立）
        return np.random.normal(0, np.sqrt(variance), 3)
    
    def step(self):
        # 计算光学力和扭矩
        optical_force = self.optical_trap.get_force(self.particle.position)
        optical_torque = self.optical_trap.get_torque(self.particle)
        
        # 添加力的监控，超过阈值时发出警告
        max_force = 1e-6  # 设置一个参考阈值
        if np.any(np.abs(optical_force) > max_force):
            print(f"警告：光学力 {optical_force} N 超过参考值 {max_force} N")
        
        # 计算阻尼力
        damping_force = -self.gamma * self.particle.velocity
        if np.any(np.abs(damping_force) > max_force):
            print(f"警告：阻尼力 {damping_force} N 超过参考值 {max_force} N")
        
        # 计算随机涨落力
        fluctuation_force = self.calculate_fluctuation_force()
        if np.any(np.abs(fluctuation_force) > max_force/10):
            print(f"警告：随机力 {fluctuation_force} N 超过参考值 {max_force/10} N")
        
        # 总力 = 光学力 + 阻尼力 + 随机力
        total_force = optical_force + damping_force + fluctuation_force
        
        # 监控总力
        if np.any(np.abs(total_force) > max_force):
            print(f"警告：总力 {total_force} N 超过参考值 {max_force} N")
        
        # 更新加速度
        self.particle.acceleration = total_force / self.particle.mass
        
        # 更新速度
        self.particle.velocity += self.particle.acceleration * self.timestep
        
        # 更新位置
        self.particle.position += self.particle.velocity * self.timestep
        
        # 计算角阻尼（减小为线性阻尼的1/10）
        # 计算角阻尼
        if self.environment.medium == 'liquid':
            # 液体环境：γrot = 8πa³η
            gamma_rot = 8 * np.pi * self.particle.radius**3 * self.environment.eta
        else:
            # 气体环境：γrot = 8πa³η * (P_gas/101325)
            gamma_rot = 8 * np.pi * self.particle.radius**3 * self.environment.eta * (self.environment.P_gas/101325)
        
        angular_damping = -gamma_rot * self.particle.angular_velocity
        
        # 监控角阻尼扭矩
        max_damping_torque = 1e-15  # 参考最大角阻尼扭矩
        if np.any(np.abs(angular_damping) > max_damping_torque):
            print(f"警告：角阻尼扭矩 {angular_damping} N·m 超过参考值 {max_damping_torque} N·m")
        
        # 监控光学扭矩
        max_optical_torque = 1e-15  # 参考最大光学扭矩
        if np.any(np.abs(optical_torque) > max_optical_torque):
            print(f"警告：光学扭矩 {optical_torque} N·m 超过参考值 {max_optical_torque} N·m")
        
        # 总扭矩 = 光学扭矩 + 角阻尼
        total_torque = optical_torque + angular_damping
        
        # 监控总扭矩
        max_total_torque = 1e-15  # 参考最大总扭矩
        if np.any(np.abs(total_torque) > max_total_torque):
            print(f"警告：总扭矩 {total_torque} N·m 超过参考值 {max_total_torque} N·m")
        
        # 更新角加速度
        self.particle.angular_acceleration = total_torque / self.particle.moment_of_inertia
        
        # 监控角加速度
        max_ang_acc = 1e9
        if np.any(np.abs(self.particle.angular_acceleration) > max_ang_acc):
            print(f"警告：角加速度 {self.particle.angular_acceleration} rad/s² 超过参考值 {max_ang_acc} rad/s²")
        
        # 更新角速度
        self.particle.angular_velocity += self.particle.angular_acceleration * self.timestep
        
        # 监控角速度
        max_ang_vel = 1e7
        if np.any(np.abs(self.particle.angular_velocity) > max_ang_vel):
            print(f"警告：角速度 {self.particle.angular_velocity} rad/s 超过参考值 {max_ang_vel} rad/s")
        
        # 计算角速度引起的切向速度
        dx = self.particle.position[0] - self.optical_trap.center[0]
        dy = self.particle.position[1] - self.optical_trap.center[1]
        r = np.sqrt(dx**2 + dy**2)
        
        if r > 0:
            # 切向单位向量
            tangential_dir = np.array([-dy/r, dx/r, 0])
            # 切向速度 = 角速度 * 半径
            tangential_velocity = np.linalg.norm(self.particle.angular_velocity) * r * tangential_dir
            # 更新位置时加入切向运动的贡献
            self.particle.position += tangential_velocity * self.timestep
        
        # 保存当前受力和扭矩
        self.particle.force = total_force
        self.particle.torque = total_torque
        
        # 更新时间
        self.time += self.timestep
        
        # 记录轨迹和状态
        self.trajectory.append((self.time, self.particle.position.copy()))
        self.velocity_history.append(self.particle.velocity.copy())
        self.force_history.append(total_force.copy())
        self.angular_trajectory.append((self.time, self.particle.angular_velocity.copy()))
        self.torque_history.append(total_torque.copy())  # 添加力矩历史记录
        return self.particle.position
    
    def simulate(self, duration, timestep=None):
        """运行模拟指定时间"""
        if timestep is not None:
            self.timestep = timestep
        
        num_steps = int(duration / self.timestep)
        # 重置历史记录
        self.trajectory = []
        self.velocity_history = []
        self.force_history = []
        self.angular_trajectory = []  # 存储角速度历史
        self.torque_history = []  # 添加力矩历史的重置
        
        for _ in range(num_steps):
            self.step()
        
        return self.get_trajectory()
    
    def get_oscillation_frequency(self):
        """计算粒子在光阱中的自然振荡频率"""
        return np.sqrt(self.optical_trap.kappa / self.particle.mass)
    
    def get_quality_factor(self):
        """计算品质因子 Q = ω / Γ"""
        omega = self.get_oscillation_frequency()
        Gamma = self.gamma / self.particle.mass
        return omega / Gamma
    
    def get_damping_regime(self):
        """判断阻尼状态"""
        omega = self.get_oscillation_frequency()[0]  # 使用x方向的值
        Gamma = self.gamma[0] / self.particle.mass
        
        if Gamma > 2 * omega:
            return "overdamped"
        elif Gamma < 2 * omega:
            return "underdamped"
        else:
            return "critically damped"
    
    def get_trajectory(self):
        """返回粒子轨迹数据"""
        times = [t for t, pos in self.trajectory]
        positions = np.array([pos for t, pos in self.trajectory])
        velocities = np.array([vel for vel in self.velocity_history])
        forces = np.array([force for force in self.force_history])
        angular_velocities = np.array([ang_vel for t, ang_vel in self.angular_trajectory])
        torques = np.array([torque for torque in self.torque_history])  # 新增力矩历史
        
        return {
            'time': np.array(times),
            'position': positions,
            'velocity': velocities,
            'force': forces,
            'angular_velocity': angular_velocities,
            'torque': torques  # 返回力矩数据
        }
    
    def save_trajectory_to_csv(self, filename):
        """将轨迹数据保存到CSV文件"""
        data = self.get_trajectory()
        
        with open(filename, 'w') as f:
            # 写入表头
            f.write("Time (s),X (m),Y (m),Z (m),Vx (m/s),Vy (m/s),Vz (m/s),Fx (N),Fy (N),Fz (N),ωx (rad/s),ωy (rad/s),ωz (rad/s),τx (N·m),τy (N·m),τz (N·m)\n")
            
            # 写入数据
            for i in range(len(data['time'])):
                t = data['time'][i]
                x, y, z = data['position'][i]
                vx, vy, vz = data['velocity'][i]
                fx, fy, fz = data['force'][i]
                ωx, ωy, ωz = data['angular_velocity'][i]
                τx, τy, τz = data['torque'][i]
                
                f.write(f"{t:.6e},{x:.6e},{y:.6e},{z:.6e},{vx:.6e},{vy:.6e},{vz:.6e},{fx:.6e},{fy:.6e},{fz:.6e},{ωx:.6e},{ωy:.6e},{ωz:.6e},{τx:.6e},{τy:.6e},{τz:.6e}\n")

    def animate_trajectory(self, duration=1e-4, timestep=1e-7):
        """实时动画显示粒子运动"""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import numpy as np
        
        # 创建一个图形窗口，包含三个子图
        fig = plt.figure(figsize=(12, 8))
        
        ax1 = fig.add_subplot(121)  # x-y平面图
        ax2 = fig.add_subplot(222)  # 位置随时间变化图
        ax3 = fig.add_subplot(224)  # 角速度随时间变化图
        
        # 设置x-y平面图的范围
        limit = 5e-6
        ax1.set_xlim([-limit, limit])
        ax1.set_ylim([-limit, limit])
        
        # 添加光场颜色图
        x = np.linspace(-2e-6, 2e-6, 200)
        y = np.linspace(-2e-6, 2e-6, 200)
        X, Y = np.meshgrid(x, y)
        
        field_2d = self.optical_trap.get_field_2d(X, Y)
        if field_2d is not None:
            im = ax1.pcolormesh(X, Y, field_2d,
                              cmap='YlOrBr',
                              alpha=0.3,
                              shading='auto')
            plt.colorbar(im, ax=ax1, label='Field Intensity')
        
        # 初始化轨迹线
        line_xy, = ax1.plot([], [], 'b-', label='Trajectory')
        point, = ax1.plot([], [], 'ro', markersize=8, label='Particle')
        
        line_x, = ax2.plot([], [], 'r-', label='X')
        line_y, = ax2.plot([], [], 'g-', label='Y')
        line_z, = ax2.plot([], [], 'b-', label='Z')
        
        line_w, = ax3.plot([], [], 'b-', label='Angular Speed')
        
        # 设置标签
        ax1.set_title('Particle Trajectory in X-Y Plane')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.grid(True)
        ax1.axis('equal')
        ax1.legend()
        
        ax2.set_title('Position vs. Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position (m)')
        ax2.grid(True)
        ax2.legend()
        
        ax3.set_title('Angular Speed vs. Time')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Angular Speed (rad/s)')
        ax3.grid(True)
        ax3.legend()
        
        def init():
            line_xy.set_data([], [])
            point.set_data([], [])
            line_x.set_data([], [])
            line_y.set_data([], [])
            line_z.set_data([], [])
            line_w.set_data([], [])
            return line_xy, point, line_x, line_y, line_z, line_w
        
        def update(frame):
            # 执行一步模拟
            self.step()
            data = self.get_trajectory()
            positions = data['position']
            angular_velocities = data['angular_velocity']
            times = data['time']
            
            # 计算角速度大小
            angular_speed = np.linalg.norm(angular_velocities, axis=1)
            
            # 更新x-y平面轨迹
            line_xy.set_data(positions[:, 0], positions[:, 1])
            point.set_data([positions[-1, 0]], [positions[-1, 1]])
            
            # 更新位置随时间变化
            line_x.set_data(times, positions[:, 0])
            line_y.set_data(times, positions[:, 1])
            line_z.set_data(times, positions[:, 2])
            
            # 更新角速度随时间变化
            line_w.set_data(times, angular_speed)
            
            # 自动调整坐标范围
            ax2.relim()
            ax2.autoscale_view()
            ax3.relim()
            ax3.autoscale_view()
            
            return line_xy, point, line_x, line_y, line_z, line_w
        
        # 计算帧数
        frames = int(duration / timestep)
        
        # 创建动画，显示间隔改为0.8ms以加快5倍速度
        anim = FuncAnimation(fig, update, frames=frames,
                           init_func=init, interval=0.8, blit=True)
        plt.tight_layout()
        plt.show()

    def animate_trajectory_3d(self, duration=1e-4, timestep=1e-7, view_angles=None):
        """在三维空间中展示粒子轨迹的动画
        
        参数:
        duration (float): 模拟总时长 (s)
        timestep (float): 时间步长 (s)
        view_angles (tuple): 视角的仰角和方位角，默认为 (30, 45)
        """
        if view_angles is None:
            view_angles = (30, 45)
        
        # 运行模拟并记录轨迹
        self.simulate(duration, timestep)
        
        # 创建图形和3D坐标系
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 提取轨迹数据
        times = np.array([t for t, _ in self.trajectory])
        positions = np.array([p for _, p in self.trajectory])
        
        # 设置坐标轴范围
        margin = 0.5e-6  # 添加一些边距
        x_min, x_max = positions[:, 0].min() - margin, positions[:, 0].max() + margin
        y_min, y_max = positions[:, 1].min() - margin, positions[:, 1].max() + margin
        z_min, z_max = positions[:, 2].min() - margin, positions[:, 2].max() + margin
        
        def update(frame):
            ax.clear()
            
            # 绘制完整轨迹（半透明）
            ax.plot(positions[:frame, 0], positions[:frame, 1], positions[:frame, 2],
                    'b-', alpha=0.3, label='Trajectory')
            
            # 绘制当前位置（实心点）
            if frame > 0:
                ax.scatter(positions[frame-1, 0], positions[frame-1, 1], positions[frame-1, 2],
                        c='red', marker='o', s=100, label='Particle')
            
            # 绘制光阱中心
            ax.scatter([self.optical_trap.center[0]], [self.optical_trap.center[1]],
                    [self.optical_trap.center[2]], c='green', marker='+',
                    s=100, label='Trap Center')
            
            # 设置标签和标题
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Time: {times[frame]:.2e} s')
            
            # 设置坐标轴范围
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            
            # 设置视角
            ax.view_init(elev=view_angles[0], azim=view_angles[1])
            
            # 添加图例
            ax.legend()
        
        # 创建动画
        frames = len(times)
        anim = animation.FuncAnimation(fig, update, frames=frames,
                                    interval=0.5, blit=False)  # 将 interval 从 50 减小到 0.5，大幅提高动画速度
        
        # 显示动画
        plt.show()    



def plot_3d_intensity(self, num_points=50, alpha=0.5):
    """绘制光阱的三维强度分布图，使用交互式切片平面
    
    参数:
    num_points (int): 每个维度上的采样点数，默认50
    alpha (float): 散点图的透明度，默认0.5
    """
    if self.field is None:
        print("错误：光场未初始化，请先调用 set_field 方法设置光场")
        return
        
    # 创建图形窗口
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 创建采样点
    x = np.linspace(self.grid_x[0], self.grid_x[-1], num_points)
    y = np.linspace(self.grid_y[0], self.grid_y[-1], num_points)
    z = np.linspace(self.grid_z[0], self.grid_z[-1], num_points)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 对光场进行插值
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (self.grid_x, self.grid_y, self.grid_z),
        self.field
    )
    
    # 计算插值后的光场强度
    points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    intensity = interp(points).reshape(X.shape)
    
    # 归一化强度值
    intensity_norm = intensity / np.max(intensity)
    
    # 创建三个正交平面的切片
    mid_x = num_points // 2
    mid_y = num_points // 2
    mid_z = num_points // 2
    
    # XY平面
    xy_slice = ax.contourf(X[:, :, mid_z], Y[:, :, mid_z], 
                          intensity_norm[:, :, mid_z],
                          zdir='z', offset=z[mid_z],
                          cmap='viridis', alpha=alpha)
    
    # XZ平面
    xz_slice = ax.contourf(X[:, mid_y, :], Z[:, mid_y, :],
                          intensity_norm[:, mid_y, :],
                          zdir='y', offset=y[mid_y],
                          cmap='viridis', alpha=alpha)
    
    # YZ平面
    yz_slice = ax.contourf(Y[mid_x, :, :], Z[mid_x, :, :],
                          intensity_norm[mid_x, :, :],
                          zdir='x', offset=x[mid_x],
                          cmap='viridis', alpha=alpha)
    
    # 添加光阱中心标记
    ax.scatter([self.center[0]], [self.center[1]], [self.center[2]],
              color='red', marker='o', s=100, label='Trap Center')
    
    # 设置标签和标题
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Optical Trap Intensity Distribution')
    
    # 添加颜色条
    plt.colorbar(xy_slice, label='Normalized Intensity')
    
    # 设置坐标轴范围
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([y[0], y[-1]])
    ax.set_zlim([z[0], z[-1]])
    
    # 添加图例
    ax.legend()
    
    # 设置视角
    ax.view_init(elev=30, azim=45)
    
    # 调整布局并显示
    plt.tight_layout()
    plt.show()


