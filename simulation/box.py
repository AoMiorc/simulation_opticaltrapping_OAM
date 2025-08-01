import numpy as np
from scipy.constants import Boltzmann as k_B

class SimulationBox:
    """三维模拟画布，整合粒子、光阱和环境
    3D simulation canvas that integrates particles, optical traps and environment"""
    
    def __init__(self, particles=None, environment=None, optical_trap=None, timestep=1e-6):
        """初始化模拟盒子 / Initialize simulation box
        
        参数 / Parameters:
            particles: 单个粒子对象或粒子对象列表
            environment: 环境对象
            optical_trap: 光阱对象
            timestep: 时间步长 (s)，默认1μs
        """
        self.environment = environment
        self.optical_trap = optical_trap
        self.timestep = timestep
        self.time = 0.0
        
        if particles is None:
            self.particles = []
        elif isinstance(particles, (list, tuple)):
            self.particles = list(particles)
        else:
            self.particles = [particles]
            
    def update_damping(self):
        """更新阻尼系数（如果环境参数变化）
        Update damping coefficient (if environment parameters change)"""
        self.gamma = np.array([
            self.environment.get_drag_coefficient(self.particle),
            self.environment.get_drag_coefficient(self.particle),
            self.environment.get_drag_coefficient(self.particle)
        ])
    
    def _step(self, particle_index=None):
        """更新粒子状态（内部方法）/ Update particle state (internal method)
        
        参数 / Parameters:
            particle_index: 要更新的粒子索引，如果为None则更新所有粒子
                          Index of particle to update, if None update all particles
        """
        # 确定要更新的粒子
        if particle_index is None:
            # 如果没有指定索引，更新所有粒子
            particles_to_update = self.particles
        else:
            # 如果指定了索引，只更新该粒子
            if 0 <= particle_index < len(self.particles):
                particles_to_update = [self.particles[particle_index]]
            else:
                print(f"警告：粒子索引 {particle_index} 超出范围，将更新所有粒子")
                print(f"Warning: Particle index {particle_index} out of range, updating all particles")
                particles_to_update = self.particles
        
        # 更新指定的粒子
        for particle in particles_to_update:
            # 计算光学力
            optical_force = self.optical_trap.get_force(particle.position)
            
            # 计算当前粒子的阻尼系数
            gamma = self.environment.get_drag_coefficient(particle)
            
            # 计算随机涨落力
            variance = 2 * gamma * k_B * self.environment.T / self.timestep
            fluctuation_force = np.random.normal(0, np.sqrt(variance), 3)
            
            # 使用半隐式欧拉方法处理阻尼
            # v_{n+1} = (v_n + (F_optical + F_random) * dt / m) / (1 + γ * dt / m)
            non_damping_force = optical_force + fluctuation_force
            
            # 更新速度（半隐式处理阻尼）
            damping_factor = 1 + gamma * self.timestep / particle.mass
            velocity_increment = non_damping_force * self.timestep / particle.mass
            particle.velocity = (particle.velocity + velocity_increment) / damping_factor
            
            # 计算光学扭矩
            # 计算光学扭矩
            optical_torque = self.optical_trap.calculate_torque_at_position(particle.position)
            torque_z = optical_torque[2]  # 使用扭矩的 z 分量，包括符号
            
            # 确定旋转中心
            if self.optical_trap.axis_points is None or self.optical_trap.axis_direction is None:
                self.optical_trap.calculate_angular_momentum_axis()
            
            if self.optical_trap.axis_points is not None and len(self.optical_trap.axis_points) > 1:
                axis_start = self.optical_trap.axis_points[0]
                axis_direction = self.optical_trap.axis_direction
                particle_vec = particle.position - axis_start
                projection_length = np.dot(particle_vec, axis_direction)
                rotation_center = axis_start + projection_length * axis_direction
            else:
                rotation_center = self.optical_trap.center
            
            dx = particle.position[0] - rotation_center[0]
            dy = particle.position[1] - rotation_center[1]
            r = np.sqrt(dx**2 + dy**2)
            
            if r > 0:
                tangential_dir = np.array([-dy/r, dx/r, 0])
                v_tangential = np.dot(particle.velocity, tangential_dir)
                omega_magnitude = abs(v_tangential) / r
                omega_direction = 1 if v_tangential > 0 else -1
                particle.angular_velocity = np.array([0, 0, omega_magnitude * omega_direction])
            else:
                particle.angular_velocity = np.array([0.0, 0.0, 0.0])
            
            I_axis = particle.moment_of_inertia + particle.mass * r**2
            alpha_z = torque_z / I_axis
            particle.angular_acceleration = np.array([0, 0, alpha_z])
            
            if r > 0:
                tangential_acceleration_magnitude = abs(alpha_z) * r
                if alpha_z >= 0:
                    tangential_acceleration = tangential_acceleration_magnitude * tangential_dir
                else:
                    tangential_acceleration = tangential_acceleration_magnitude * (-tangential_dir)
                particle.velocity += tangential_acceleration * self.timestep
            
            # 位置更新
            particle.position += particle.velocity * self.timestep
            
            # 修正：移除重复的阻尼项
            total_force = optical_force + fluctuation_force  # 删除 - gamma * particle.velocity
            
            particle.acceleration = total_force / particle.mass
            
            # 更新时间
            self.time += self.timestep
            
            # 记录轨迹和状态 - 修改为多粒子版本
            particle_idx = self.particles.index(particle)
            self.trajectory[particle_idx].append((self.time, particle.position.copy()))
            self.velocity_history[particle_idx].append(particle.velocity.copy())
            self.force_history[particle_idx].append(total_force.copy())
            self.angular_trajectory[particle_idx].append((self.time, particle.angular_velocity.copy()))
            self.torque_history[particle_idx].append(optical_torque.copy())
        
        # 返回更新的粒子位置（如果只更新一个粒子则返回该粒子位置，否则返回所有粒子位置）
        if particle_index is not None and 0 <= particle_index < len(self.particles):
            return self.particles[particle_index].position
        else:
            return [particle.position for particle in self.particles]
    
    def simulate(self, duration, save_interval=None, show_progress=True):
        """运行模拟 / Run simulation
        
        参数 / Parameters:
            duration: 模拟时长 / Simulation duration
            save_interval: 保存间隔 / Save interval
            show_progress: 是否显示进度条 / Whether to show progress bar
        """
        num_steps = int(duration / self.timestep)
        # 重置历史记录 - 改为多粒子版本
        self.trajectory = [[] for _ in self.particles]  # 每个粒子一个轨迹列表
        self.velocity_history = [[] for _ in self.particles]
        self.force_history = [[] for _ in self.particles]
        self.angular_trajectory = [[] for _ in self.particles]
        self.torque_history = [[] for _ in self.particles]
        
        # 进度条设置
        if show_progress:
            print(f"开始模拟: {num_steps} 步, 时间步长: {self.timestep:.2e}s")
            print(f"预计模拟时长: {duration}s")
            progress_interval = max(1, num_steps // 100)  # 每1%更新一次
        
        for step in range(num_steps):
            self._step()
            
            # 显示进度
            if show_progress and (step + 1) % progress_interval == 0:
                progress = (step + 1) / num_steps * 100
                bar_length = 50
                filled_length = int(bar_length * (step + 1) // num_steps)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f'\r进度: |{bar}| {progress:.1f}% ({step + 1}/{num_steps} 步)', end='', flush=True)
        
        if show_progress:
            print('\n模拟完成!')
        
        return self.get_trajectory()
    
    
    def get_trajectory(self):
        """获取所有粒子的轨迹数据 / Get trajectory data for all particles"""
        trajectories = []
        
        for i, particle in enumerate(self.particles):
            times = [t for t, pos in self.trajectory[i]]
            positions = np.array([pos for t, pos in self.trajectory[i]])
            velocities = np.array(self.velocity_history[i])
            forces = np.array(self.force_history[i])
            angular_velocities = np.array([ang_vel for t, ang_vel in self.angular_trajectory[i]])
            torques = np.array(self.torque_history[i])
            
            trajectories.append({
                'time': np.array(times),
                'position': positions,
                'velocity': velocities,
                'force': forces,
                'angular_velocity': angular_velocities,
                'torque': torques
            })
        
        return trajectories
    
    def save_trajectory_to_csv(self, filename):
        """将多粒子轨迹数据保存到CSV文件
        Save multi-particle trajectory data to CSV file"""
        trajectories = self.get_trajectory()  # 获取所有粒子的轨迹数据
        
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            # 写入表头 - 修改扭矩单位
            f.write("Particle_ID,Time (s),X (m),Y (m),Z (m),Vx (m/s),Vy (m/s),Vz (m/s),Fx (N),Fy (N),Fz (N),ωx (rad/s),ωy (rad/s),ωz (rad/s),τx (pN·μm),τy (pN·μm),τz (pN·μm)\n")
            
            # 写入每个粒子的数据
            for particle_id, data in enumerate(trajectories):
                for i in range(len(data['time'])):
                    t = data['time'][i]
                    x, y, z = data['position'][i]
                    vx, vy, vz = data['velocity'][i]
                    fx, fy, fz = data['force'][i]
                    ωx, ωy, ωz = data['angular_velocity'][i]
                    τx, τy, τz = data['torque'][i]
                    
                    # 转换扭矩单位：N⋅m → pN⋅μm (乘以 10^18)
                    τx_pN_um = τx * 1e18
                    τy_pN_um = τy * 1e18
                    τz_pN_um = τz * 1e18
                    
                    f.write(f"{particle_id},{t:.6e},{x:.6e},{y:.6e},{z:.6e},{vx:.6e},{vy:.6e},{vz:.6e},{fx:.6e},{fy:.6e},{fz:.6e},{ωx:.6e},{ωy:.6e},{ωz:.6e},{τx_pN_um:.6e},{τy_pN_um:.6e},{τz_pN_um:.6e}\n")