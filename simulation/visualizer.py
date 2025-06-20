import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class TrajectoryVisualizer:
    """从CSV文件读取多粒子轨迹数据并进行可视化的类 / Class for reading multi-particle trajectory data from CSV files and visualization"""
    
    def __init__(self, csv_file=None):
        """初始化可视化器 / Initialize the visualizer
        
        Args:
            csv_file: CSV文件路径 / CSV file path
        """
        self.data = None
        self.particles_data = {}  # 存储每个粒子的数据 / Store data for each particle
        self.csv_file = csv_file
        if csv_file:
            self.load_data(csv_file)
    
    def load_data(self, csv_file):
        """从CSV文件加载多粒子轨迹数据 / Load multi-particle trajectory data from CSV file
        
        Args:
            csv_file: CSV文件路径 / CSV file path
        """
        try:
            self.data = pd.read_csv(csv_file)
            self.csv_file = csv_file
            
            # 按粒子ID分组数据 / Group data by particle ID
            if 'Particle_ID' in self.data.columns:
                self.particles_data = {}
                for particle_id in self.data['Particle_ID'].unique():
                    self.particles_data[particle_id] = self.data[self.data['Particle_ID'] == particle_id].copy()
                print(f"成功加载数据，共 {len(self.particles_data)} 个粒子，{len(self.data)} 个数据点 / Successfully loaded data with {len(self.particles_data)} particles and {len(self.data)} data points")
            else:
                # 兼容单粒子格式 / Compatible with single particle format
                self.particles_data = {0: self.data}
                print(f"成功加载单粒子数据，共 {len(self.data)} 个数据点 / Successfully loaded single particle data with {len(self.data)} data points")
                
        except Exception as e:
            print(f"加载CSV文件失败: {e} / Failed to load CSV file: {e}")
            self.data = None
            self.particles_data = {}
    
    def plot_2d_trajectory(self, plane='xy', figsize=(10, 8), particle_ids=None):
        """绘制多粒子2D轨迹图 / Plot multi-particle 2D trajectory
        
        Args:
            plane: 投影平面 ('xy', 'xz', 'yz') / Projection plane ('xy', 'xz', 'yz')
            figsize: 图形尺寸 / Figure size
            particle_ids: 要绘制的粒子ID列表，None表示绘制所有粒子 / List of particle IDs to plot, None for all particles
        """
        if not self.particles_data:
            print("请先加载数据 / Please load data first")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 确定要绘制的粒子 / Determine particles to plot
        if particle_ids is None:
            particle_ids = list(self.particles_data.keys())
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(particle_ids)))
        
        for i, particle_id in enumerate(particle_ids):
            if particle_id not in self.particles_data:
                continue
                
            data = self.particles_data[particle_id]
            
            if plane == 'xy':
                ax.plot(data['X (m)'], data['Y (m)'], color=colors[i], 
                       linewidth=1, label=f'粒子 {particle_id} / Particle {particle_id}')
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title('多粒子轨迹 (XY平面) / Multi-particle Trajectory (XY Plane)')
            elif plane == 'xz':
                ax.plot(data['X (m)'], data['Z (m)'], color=colors[i], 
                       linewidth=1, label=f'粒子 {particle_id} / Particle {particle_id}')
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Z (m)')
                ax.set_title('多粒子轨迹 (XZ平面) / Multi-particle Trajectory (XZ Plane)')
            elif plane == 'yz':
                ax.plot(data['Y (m)'], data['Z (m)'], color=colors[i], 
                       linewidth=1, label=f'粒子 {particle_id} / Particle {particle_id}')
                ax.set_xlabel('Y (m)')
                ax.set_ylabel('Z (m)')
                ax.set_title('多粒子轨迹 (YZ平面) / Multi-particle Trajectory (YZ Plane)')
        
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_3d_trajectory(self, figsize=(12, 9), particle_ids=None):
        """绘制多粒子3D轨迹图 / Plot multi-particle 3D trajectory
        
        Args:
            figsize: 图形尺寸 / Figure size
            particle_ids: 要绘制的粒子ID列表，None表示绘制所有粒子 / List of particle IDs to plot, None for all particles
        """
        if not self.particles_data:
            print("请先加载数据 / Please load data first")
            return
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 确定要绘制的粒子 / Determine particles to plot
        if particle_ids is None:
            particle_ids = list(self.particles_data.keys())
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(particle_ids)))
        
        for i, particle_id in enumerate(particle_ids):
            if particle_id not in self.particles_data:
                continue
                
            data = self.particles_data[particle_id]
            
            ax.plot(data['X (m)'], data['Y (m)'], data['Z (m)'], 
                   color=colors[i], linewidth=1, alpha=0.7, label=f'粒子 {particle_id} / Particle {particle_id}')
            
            # 标记起点和终点 / Mark start and end points
            ax.scatter(data['X (m)'].iloc[0], data['Y (m)'].iloc[0], 
                      data['Z (m)'].iloc[0], color=colors[i], s=50, marker='o')
            ax.scatter(data['X (m)'].iloc[-1], data['Y (m)'].iloc[-1], 
                      data['Z (m)'].iloc[-1], color=colors[i], s=50, marker='s')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('多粒子3D轨迹 / Multi-particle 3D Trajectory')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_velocity_magnitude(self, figsize=(12, 6), particle_ids=None):
        """绘制多粒子速度和角速度大小随时间变化 / Plot velocity and angular velocity magnitude vs time for multiple particles
        
        Args:
            figsize: 图形尺寸 / Figure size
            particle_ids: 要绘制的粒子ID列表，None表示绘制所有粒子 / List of particle IDs to plot, None for all particles
        """
        if not self.particles_data:
            print("请先加载数据 / Please load data first")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 确定要绘制的粒子 / Determine particles to plot
        if particle_ids is None:
            particle_ids = list(self.particles_data.keys())
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(particle_ids)))
        
        for i, particle_id in enumerate(particle_ids):
            if particle_id not in self.particles_data:
                continue
                
            data = self.particles_data[particle_id]
            
            # 速度大小 / Velocity magnitude
            v_magnitude = np.sqrt(data['Vx (m/s)']**2 + 
                                 data['Vy (m/s)']**2 + 
                                 data['Vz (m/s)']**2)
            axes[0].plot(data['Time (s)'], v_magnitude, color=colors[i], 
                        linewidth=2, label=f'粒子 {particle_id} / Particle {particle_id}')
            
            # 角速度大小 / Angular velocity magnitude
            omega_magnitude = np.sqrt(data['ωx (rad/s)']**2 + 
                                     data['ωy (rad/s)']**2 + 
                                     data['ωz (rad/s)']**2)
            axes[1].plot(data['Time (s)'], omega_magnitude, color=colors[i], 
                        linewidth=2, label=f'粒子 {particle_id} / Particle {particle_id}')
        
        axes[0].set_xlabel('时间 (s) / Time (s)')
        axes[0].set_ylabel('速度大小 (m/s) / Velocity Magnitude (m/s)')
        axes[0].set_title('线速度大小 / Linear Velocity Magnitude')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].set_xlabel('时间 (s) / Time (s)')
        axes[1].set_ylabel('角速度大小 (rad/s) / Angular Velocity Magnitude (rad/s)')
        axes[1].set_title('角速度大小 / Angular Velocity Magnitude')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_force_magnitude(self, figsize=(12, 6), particle_ids=None):
        """绘制多粒子力和扭矩大小随时间变化 / Plot force and torque magnitude vs time for multiple particles
        
        Args:
            figsize: 图形尺寸 / Figure size
            particle_ids: 要绘制的粒子ID列表，None表示绘制所有粒子 / List of particle IDs to plot, None for all particles
        """
        if not self.particles_data:
            print("请先加载数据 / Please load data first")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 确定要绘制的粒子 / Determine particles to plot
        if particle_ids is None:
            particle_ids = list(self.particles_data.keys())
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(particle_ids)))
        
        for i, particle_id in enumerate(particle_ids):
            if particle_id not in self.particles_data:
                continue
                
            data = self.particles_data[particle_id]
            
            # 力大小 / Force magnitude
            f_magnitude = np.sqrt(data['Fx (N)']**2 + 
                                 data['Fy (N)']**2 + 
                                 data['Fz (N)']**2)
            axes[0].plot(data['Time (s)'], f_magnitude, color=colors[i], 
                        linewidth=2, label=f'粒子 {particle_id} / Particle {particle_id}')
            
            # 扭矩大小 / Torque magnitude
            tau_magnitude = np.sqrt(data['τx (N·m)']**2 + 
                                   data['τy (N·m)']**2 + 
                                   data['τz (N·m)']**2)
            axes[1].plot(data['Time (s)'], tau_magnitude, color=colors[i], 
                        linewidth=2, label=f'粒子 {particle_id} / Particle {particle_id}')
        
        axes[0].set_xlabel('时间 (s) / Time (s)')
        axes[0].set_ylabel('力大小 (N) / Force Magnitude (N)')
        axes[0].set_title('力大小 / Force Magnitude')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].set_xlabel('时间 (s) / Time (s)')
        axes[1].set_ylabel('扭矩大小 (N·m) / Torque Magnitude (N·m)')
        axes[1].set_title('扭矩大小 / Torque Magnitude')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_all_magnitudes(self, figsize=(15, 10), particle_ids=None):
        """绘制所有物理量的大小随时间变化 / Plot all physical quantities magnitude vs time
        
        Args:
            figsize: 图形尺寸 / Figure size
            particle_ids: 要绘制的粒子ID列表，None表示绘制所有粒子 / List of particle IDs to plot, None for all particles
        """
        if not self.particles_data:
            print("请先加载数据 / Please load data first")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # 确定要绘制的粒子 / Determine particles to plot
        if particle_ids is None:
            particle_ids = list(self.particles_data.keys())
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(particle_ids)))
        
        for i, particle_id in enumerate(particle_ids):
            if particle_id not in self.particles_data:
                continue
                
            data = self.particles_data[particle_id]
            
            # 速度大小 / Velocity magnitude
            v_magnitude = np.sqrt(data['Vx (m/s)']**2 + 
                                 data['Vy (m/s)']**2 + 
                                 data['Vz (m/s)']**2)
            axes[0].plot(data['Time (s)'], v_magnitude, color=colors[i], 
                        linewidth=2, label=f'粒子 {particle_id} / Particle {particle_id}')
            
            # 角速度大小 / Angular velocity magnitude
            omega_magnitude = np.sqrt(data['ωx (rad/s)']**2 + 
                                     data['ωy (rad/s)']**2 + 
                                     data['ωz (rad/s)']**2)
            axes[1].plot(data['Time (s)'], omega_magnitude, color=colors[i], 
                        linewidth=2, label=f'粒子 {particle_id} / Particle {particle_id}')
            
            # 力大小 / Force magnitude
            f_magnitude = np.sqrt(data['Fx (N)']**2 + 
                                 data['Fy (N)']**2 + 
                                 data['Fz (N)']**2)
            axes[2].plot(data['Time (s)'], f_magnitude, color=colors[i], 
                        linewidth=2, label=f'粒子 {particle_id} / Particle {particle_id}')
            
            # 扭矩大小 / Torque magnitude
            tau_magnitude = np.sqrt(data['τx (N·m)']**2 + 
                                   data['τy (N·m)']**2 + 
                                   data['τz (N·m)']**2)
            axes[3].plot(data['Time (s)'], tau_magnitude, color=colors[i], 
                        linewidth=2, label=f'粒子 {particle_id} / Particle {particle_id}')
        
        # 设置子图标题和标签 / Set subplot titles and labels
        titles = ['线速度大小 / Linear Velocity Magnitude', '角速度大小 / Angular Velocity Magnitude', 
                 '力大小 / Force Magnitude', '扭矩大小 / Torque Magnitude']
        ylabels = ['速度 (m/s) / Velocity (m/s)', '角速度 (rad/s) / Angular Velocity (rad/s)', 
                  '力 (N) / Force (N)', '扭矩 (N·m) / Torque (N·m)']
        
        for j, (title, ylabel) in enumerate(zip(titles, ylabels)):
            axes[j].set_xlabel('时间 (s) / Time (s)')
            axes[j].set_ylabel(ylabel)
            axes[j].set_title(title)
            axes[j].grid(True, alpha=0.3)
            axes[j].legend()
        
        plt.tight_layout()
        plt.show()
    
    def get_statistics(self):
        """获取多粒子轨迹数据的统计信息 / Get statistical information of multi-particle trajectory data"""
        if not self.particles_data:
            print("请先加载数据 / Please load data first")
            return
        
        print(f"粒子数量 / Number of particles: {len(self.particles_data)}")
        print(f"总数据点数 / Total data points: {len(self.data) if self.data is not None else 0}")
        
        for particle_id, data in self.particles_data.items():
            print(f"\n粒子 {particle_id} / Particle {particle_id}:")
            print(f"  数据点数量 / Number of data points: {len(data)}")
            print(f"  时间范围 / Time range: {data['Time (s)'].min():.2e} - {data['Time (s)'].max():.2e} s")
            
            # 计算最大速度和力 / Calculate maximum velocity and force
            v_max = np.sqrt(data['Vx (m/s)']**2 + data['Vy (m/s)']**2 + data['Vz (m/s)']**2).max()
            f_max = np.sqrt(data['Fx (N)']**2 + data['Fy (N)']**2 + data['Fz (N)']**2).max()
            
            print(f"  最大速度 / Maximum velocity: {v_max:.2e} m/s")
            print(f"  最大力 / Maximum force: {f_max:.2e} N")
    
    def plot_2d_trajectory_with_field(self, plane='xy', figsize=(12, 10), particle_ids=None, 
                                     optical_trap=None, field_alpha=0.6, field_levels=20):
        """绘制带有光场强度背景的2D轨迹图 / Plot 2D trajectory with optical field intensity background
        
        Args:
            plane: 投影平面 ('xy', 'xz', 'yz') / Projection plane ('xy', 'xz', 'yz')
            figsize: 图形尺寸 / Figure size
            particle_ids: 要绘制的粒子ID列表，None表示绘制所有粒子 / List of particle IDs to plot, None for all particles
            optical_trap: 光阱对象，用于获取场强度 / OpticalTrap object for field intensity
            field_alpha: 背景场透明度 (0-1) / Background field transparency (0-1)
            field_levels: 等高线层数 / Number of contour levels
        """
        if not self.particles_data:
            print("请先加载数据 / Please load data first")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制光场强度背景 / Plot optical field intensity background
        if optical_trap is not None and optical_trap.field is not None:
            self._plot_field_background(ax, optical_trap, plane, field_alpha, field_levels)
        
        # 确定要绘制的粒子 / Determine particles to plot
        if particle_ids is None:
            particle_ids = list(self.particles_data.keys())
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(particle_ids)))
        
        for i, particle_id in enumerate(particle_ids):
            if particle_id not in self.particles_data:
                continue
                
            data = self.particles_data[particle_id]
            
            if plane == 'xy':
                ax.plot(data['X (m)'] * 1e6, data['Y (m)'] * 1e6, color=colors[i], 
                       linewidth=2, label=f'粒子 {particle_id} / Particle {particle_id}', alpha=0.8)
                # 标记起点和终点 / Mark start and end points
                ax.scatter(data['X (m)'].iloc[0] * 1e6, data['Y (m)'].iloc[0] * 1e6, 
                          color=colors[i], s=100, marker='o', edgecolor='white', linewidth=2, zorder=10)
                ax.scatter(data['X (m)'].iloc[-1] * 1e6, data['Y (m)'].iloc[-1] * 1e6, 
                          color=colors[i], s=100, marker='s', edgecolor='white', linewidth=2, zorder=10)
                ax.set_xlabel('X (μm)')
                ax.set_ylabel('Y (μm)')
                ax.set_title('粒子轨迹与光场强度 (XY平面) / Particle Trajectory with Optical Field Intensity (XY Plane)')
            elif plane == 'xz':
                ax.plot(data['X (m)'] * 1e6, data['Z (m)'] * 1e6, color=colors[i], 
                       linewidth=2, label=f'粒子 {particle_id} / Particle {particle_id}', alpha=0.8)
                ax.scatter(data['X (m)'].iloc[0] * 1e6, data['Z (m)'].iloc[0] * 1e6, 
                          color=colors[i], s=100, marker='o', edgecolor='white', linewidth=2, zorder=10)
                ax.scatter(data['X (m)'].iloc[-1] * 1e6, data['Z (m)'].iloc[-1] * 1e6, 
                          color=colors[i], s=100, marker='s', edgecolor='white', linewidth=2, zorder=10)
                ax.set_xlabel('X (μm)')
                ax.set_ylabel('Z (μm)')
                ax.set_title('粒子轨迹与光场强度 (XZ平面) / Particle Trajectory with Optical Field Intensity (XZ Plane)')
            elif plane == 'yz':
                ax.plot(data['Y (m)'] * 1e6, data['Z (m)'] * 1e6, color=colors[i], 
                       linewidth=2, label=f'粒子 {particle_id} / Particle {particle_id}', alpha=0.8)
                ax.scatter(data['Y (m)'].iloc[0] * 1e6, data['Z (m)'].iloc[0] * 1e6, 
                          color=colors[i], s=100, marker='o', edgecolor='white', linewidth=2, zorder=10)
                ax.scatter(data['Y (m)'].iloc[-1] * 1e6, data['Z (m)'].iloc[-1] * 1e6, 
                          color=colors[i], s=100, marker='s', edgecolor='white', linewidth=2, zorder=10)
                ax.set_xlabel('Y (μm)')
                ax.set_ylabel('Z (μm)')
                ax.set_title('粒子轨迹与光场强度 (YZ平面) / Particle Trajectory with Optical Field Intensity (YZ Plane)')
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 添加颜色条 / Add colorbar
        if optical_trap is not None and optical_trap.field is not None and hasattr(ax, 'collections') and len(ax.collections) > 0:
            cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.8)
            cbar.set_label('归一化强度 / Normalized Intensity', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_field_background(self, ax, optical_trap, plane, alpha, levels):
        """绘制光场强度背景 / Plot optical field intensity background"""
        # 获取场网格和强度数据 / Get field grid and intensity data
        grid_x = optical_trap.grid_x * 1e6  # 转换为微米 / Convert to micrometers
        grid_y = optical_trap.grid_y * 1e6
        grid_z = optical_trap.grid_z * 1e6
        field = optical_trap.field
        
        if plane == 'xy':
            # 在z=0平面取切片 / Take slice at z=0 plane
            z_center_idx = len(grid_z) // 2
            field_slice = field[:, :, z_center_idx]
            X, Y = np.meshgrid(grid_x, grid_y, indexing='ij')
            
            # 绘制填充等高线 / Plot filled contours
            contour = ax.contourf(X, Y, field_slice, levels=levels, 
                                 cmap='hot', alpha=alpha, zorder=1)
            # 绘制等高线 / Plot contour lines
            ax.contour(X, Y, field_slice, levels=levels, 
                      colors='white', alpha=0.3, linewidths=0.5, zorder=2)
            
        elif plane == 'xz':
            # 在y=0平面取切片 / Take slice at y=0 plane
            y_center_idx = len(grid_y) // 2
            field_slice = field[:, y_center_idx, :]
            X, Z = np.meshgrid(grid_x, grid_z, indexing='ij')
            
            contour = ax.contourf(X, Z, field_slice, levels=levels, 
                                 cmap='hot', alpha=alpha, zorder=1)
            ax.contour(X, Z, field_slice, levels=levels, 
                      colors='white', alpha=0.3, linewidths=0.5, zorder=2)
            
        elif plane == 'yz':
            # 在x=0平面取切片 / Take slice at x=0 plane
            x_center_idx = len(grid_x) // 2
            field_slice = field[x_center_idx, :, :]
            Y, Z = np.meshgrid(grid_y, grid_z, indexing='ij')
            
            contour = ax.contourf(Y, Z, field_slice, levels=levels, 
                                 cmap='hot', alpha=alpha, zorder=1)
            ax.contour(Y, Z, field_slice, levels=levels, 
                      colors='white', alpha=0.3, linewidths=0.5, zorder=2)
        
        return contour
    