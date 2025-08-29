import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation
from matplotlib.colors import LogNorm


# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False 

class TrajectoryVisualizer:
    """
    Multi-particle trajectory visualizer
    Supports loading data from CSV files, SimulationBox objects, or OpticalTrap objects
    """
    
    def __init__(self, csv_file=None, simulation_box=None, optical_trap=None):
        """
        Initialize the visualizer
        
        Args:
            csv_file: CSV file path
            simulation_box: SimulationBox object
            optical_trap: OpticalTrap object
        """
        self.data = None
        self.particles_data = {}
        self.csv_file = csv_file
        self.simulation_box = simulation_box
        self.optical_trap = optical_trap
        
        
        if csv_file:
            self.load_data(csv_file)
        elif simulation_box:
            self.load_from_box(simulation_box)
        elif optical_trap:
            self.load_from_trap(optical_trap)
    
    def load_data(self, csv_file):
        """
        Load multi-particle trajectory data from CSV file
        
        Args:
            csv_file: CSV file path
        """
        try:
            self.data = pd.read_csv(csv_file)
            self.csv_file = csv_file
            
            # Group data by particle ID
            if 'Particle_ID' in self.data.columns:
                self.particles_data = {}
                for particle_id in self.data['Particle_ID'].unique():
                    self.particles_data[particle_id] = self.data[self.data['Particle_ID'] == particle_id].copy()
                print(f"Successfully loaded data with {len(self.particles_data)} particles and {len(self.data)} data points")
            else:
                # Compatible with single particle format
                self.particles_data = {0: self.data}
                print(f"Successfully loaded single particle data with {len(self.data)} data points")
                
        except Exception as e:
            print(f"Failed to load CSV file: {e}")
            self.data = None
            self.particles_data = {}
    
    def plot_2d_trajectory(self, plane='xy', figsize=(10, 8), particle_ids=None):
        """
        Plot multi-particle 2D trajectory
        
        Args:
            plane: Projection plane ('xy', 'xz', 'yz')
            figsize: Figure size
            particle_ids: List of particle IDs to plot, None for all particles
        """
        if not self.particles_data:
            print("Please load data first") 
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if particle_ids is None:
            particle_ids = list(self.particles_data.keys())
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(particle_ids)))
        
        for i, particle_id in enumerate(particle_ids):
            if particle_id not in self.particles_data:
                continue
                
            data = self.particles_data[particle_id]
            
            if plane == 'xy':
                ax.plot(data['X (m)'] * 1e6, data['Y (m)'] * 1e6, color=colors[i], 
                       linewidth=2, label=f'Particle {particle_id}', alpha=0.8)
                # Mark start and end points
                ax.scatter(data['X (m)'].iloc[0] * 1e6, data['Y (m)'].iloc[0] * 1e6, 
                          color=colors[i], s=100, marker='o', edgecolor='white', linewidth=2, zorder=10)
                ax.scatter(data['X (m)'].iloc[-1] * 1e6, data['Y (m)'].iloc[-1] * 1e6, 
                          color=colors[i], s=100, marker='s', edgecolor='white', linewidth=2, zorder=10)
                ax.set_xlabel('X (μm)')
                ax.set_ylabel('Y (μm)')
                ax.set_title('Multi-particle Trajectory (XY Plane)')
            elif plane == 'xz':
                ax.plot(data['X (m)'] * 1e6, data['Z (m)'] * 1e6, color=colors[i], 
                       linewidth=2, label=f'Particle {particle_id}', alpha=0.8)
                ax.scatter(data['X (m)'].iloc[0] * 1e6, data['Z (m)'].iloc[0] * 1e6, 
                          color=colors[i], s=100, marker='o', edgecolor='white', linewidth=2, zorder=10)
                ax.scatter(data['X (m)'].iloc[-1] * 1e6, data['Z (m)'].iloc[-1] * 1e6, 
                          color=colors[i], s=100, marker='s', edgecolor='white', linewidth=2, zorder=10)
                ax.set_xlabel('X (μm)')
                ax.set_ylabel('Z (μm)')
                ax.set_title('Multi-particle Trajectory (XZ Plane)')
            elif plane == 'yz':
                ax.plot(data['Y (m)'] * 1e6, data['Z (m)'] * 1e6, color=colors[i], 
                       linewidth=2, label=f'Particle {particle_id}', alpha=0.8)
                ax.scatter(data['Y (m)'].iloc[0] * 1e6, data['Z (m)'].iloc[0] * 1e6, 
                          color=colors[i], s=100, marker='o', edgecolor='white', linewidth=2, zorder=10)
                ax.scatter(data['Y (m)'].iloc[-1] * 1e6, data['Z (m)'].iloc[-1] * 1e6, 
                          color=colors[i], s=100, marker='s', edgecolor='white', linewidth=2, zorder=10)
                ax.set_xlabel('Y (μm)')
                ax.set_ylabel('Z (μm)')
                ax.set_title('Multi-particle Trajectory (YZ Plane)')
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_3d_trajectory(self, figsize=(12, 9), particle_ids=None):
        """ Plot multi-particle 3D trajectory
        
        Args:
            figsize: Figure size
            particle_ids: List of particle IDs to plot, None for all particles
        """
        if not self.particles_data:
            print("Please load data first")
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
            
            # Mark start and end points
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
        """ Plot velocity and angular velocity magnitude vs time for multiple particles
        
        Args:
            figsize: Figure size
            particle_ids: List of particle IDs to plot, None for all particles
        """
        if not self.particles_data:
            print("Please load data first")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Determine particles to plot
        if particle_ids is None:
            particle_ids = list(self.particles_data.keys())
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(particle_ids)))
        
        for i, particle_id in enumerate(particle_ids):
            if particle_id not in self.particles_data:
                continue
                
            data = self.particles_data[particle_id]
            
            # Velocity magnitude
            v_magnitude = np.sqrt(data['Vx (m/s)']**2 + 
                                 data['Vy (m/s)']**2 + 
                                 data['Vz (m/s)']**2)
            axes[0].plot(data['Time (s)'], v_magnitude, color=colors[i], 
                        linewidth=2, label=f'Particle {particle_id}')
            
            # Angular velocity magnitude
            omega_magnitude = np.sqrt(data['ωx (rad/s)']**2 + 
                                     data['ωy (rad/s)']**2 + 
                                     data['ωz (rad/s)']**2)
            axes[1].plot(data['Time (s)'], omega_magnitude, color=colors[i], 
                        linewidth=2, label=f'Particle {particle_id}')
        
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Velocity Magnitude (m/s)')
        axes[0].set_title('Linear Velocity Magnitude')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Angular Velocity Magnitude (rad/s)')
        axes[1].set_title('Angular Velocity Magnitude')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_force_magnitude(self, figsize=(12, 6), particle_ids=None):
        """ Plot force and torque magnitude vs time for multiple particles
        
        Args:
            figsize: Figure size
            particle_ids: List of particle IDs to plot, None for all particles
        """
        if not self.particles_data:
            print("Please load data first")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Determine particles to plot
        if particle_ids is None:
            particle_ids = list(self.particles_data.keys())
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(particle_ids)))
        
        for i, particle_id in enumerate(particle_ids):
            if particle_id not in self.particles_data:
                continue
                
            data = self.particles_data[particle_id]
            
            # Force magnitude
            f_magnitude = np.sqrt(data['Fx (N)']**2 + 
                                 data['Fy (N)']**2 + 
                                 data['Fz (N)']**2)
            axes[0].plot(data['Time (s)'], f_magnitude, color=colors[i], 
                        linewidth=2, label=f'Particle {particle_id}')
            
            # Torque magnitude
            tau_magnitude = np.sqrt(data['τx (pN·μm)']**2 + 
                                   data['τy (pN·μm)']**2 + 
                                   data['τz (pN·μm)']**2)
            axes[1].plot(data['Time (s)'], tau_magnitude, color=colors[i], 
                        linewidth=2, label=f'Particle {particle_id}')
        
        # Set subplot titles and labels
        titles = ['Force Magnitude', 'Torque Magnitude']
        ylabels = ['Force (N)', 'Torque (pN·μm)']
        xlabel = 'Time (s)'
        
        for j, (title, ylabel) in enumerate(zip(titles, ylabels)):
            axes[j].set_xlabel(xlabel)
            axes[j].set_ylabel(ylabel)
            axes[j].set_title(title)
            axes[j].grid(True, alpha=0.3)
            axes[j].legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_all_magnitudes(self, figsize=(15, 10), particle_ids=None):
        """ Plot all physical quantities magnitude vs time for multiple particles
        
        Args:
            figsize: Figure size
            particle_ids: List of particle IDs to plot, None for all particles
        """
        if not self.particles_data:
            print("Please load data first")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Determine particles to plot
        if particle_ids is None:
            particle_ids = list(self.particles_data.keys())
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(particle_ids)))
        
        for i, particle_id in enumerate(particle_ids):
            if particle_id not in self.particles_data:
                continue
                
            data = self.particles_data[particle_id]
            
            # Linear velocity magnitude
            v_magnitude = np.sqrt(data['Vx (m/s)']**2 + 
                                 data['Vy (m/s)']**2 + 
                                 data['Vz (m/s)']**2)
            axes[0].plot(data['Time (s)'], v_magnitude, color=colors[i], 
                        linewidth=2, label=f'Particle {particle_id}')
            
            # Angular velocity magnitude
            omega_magnitude = np.sqrt(data['ωx (rad/s)']**2 + 
                                     data['ωy (rad/s)']**2 + 
                                     data['ωz (rad/s)']**2)
            axes[1].plot(data['Time (s)'], omega_magnitude, color=colors[i], 
                        linewidth=2, label=f'Particle {particle_id}')
            
            # Force magnitude
            f_magnitude = np.sqrt(data['Fx (N)']**2 + 
                                 data['Fy (N)']**2 + 
                                 data['Fz (N)']**2)
            axes[2].plot(data['Time (s)'], f_magnitude, color=colors[i], 
                        linewidth=2, label=f'Particle {particle_id}')
            
            # Torque magnitude
            tau_magnitude = np.sqrt(data['τx (pN·μm)']**2 + 
                                   data['τy (pN·μm)']**2 + 
                                   data['τz (pN·μm)']**2)
            axes[3].plot(data['Time (s)'], tau_magnitude, color=colors[i], 
                        linewidth=2, label=f'Particle {particle_id}')
        
        # Set subplot titles and labels
        titles = ['Linear Velocity Magnitude', 'Angular Velocity Magnitude', 
                 'Force Magnitude', 'Torque Magnitude']
        ylabels = ['Velocity (m/s)', 'Angular Velocity (rad/s)', 
                  'Force (N)', 'Torque (pN·μm)']
        
        for j, (title, ylabel) in enumerate(zip(titles, ylabels)):
            axes[j].set_xlabel('时间 (s) / Time (s)')
            axes[j].set_ylabel(ylabel)
            axes[j].set_title(title)
            axes[j].grid(True, alpha=0.3)
            axes[j].legend()
        
        plt.tight_layout()
        plt.show()
    
    def get_statistics(self):
        """Get statistical information of multi-particle trajectory data"""
        if not self.particles_data:
            print("Please load data first")
            return
        
        print(f"Number of particles: {len(self.particles_data)}")
        print(f"Total data points: {len(self.data) if self.data is not None else 0}")
        
        all_statistics = {}
        
        for particle_id, data in self.particles_data.items():
            print(f"\nParticle {particle_id}:")
            print(f"   Number of data points: {len(data)}")
            print(f"   Time range: {data['Time (s)'].min():.2e} - {data['Time (s)'].max():.2e} s")
            
            # Extract position and time data
            positions = np.column_stack([data['X (m)'].values, data['Y (m)'].values, data['Z (m)'].values])
            times = data['Time (s)'].values
            
            # Calculate instantaneous velocity
            velocities = []
            angular_velocities = []
            
            for i in range(1, len(positions)):
                dt = times[i] - times[i-1]
                
                # Calculate instantaneous velocity
                velocity = (positions[i] - positions[i-1]) / dt
                speed = np.linalg.norm(velocity)
                velocities.append(speed)
                
                # Calculate angular velocity around Z-axis
                r1 = positions[i-1][:2]  # Take x,y coordinates
                r2 = positions[i][:2]
                
                # Calculate angle change
                angle1 = np.arctan2(r1[1], r1[0])
                angle2 = np.arctan2(r2[1], r2[0])
                
                # Handle angle wrap-around (-π to π)
                dangle = angle2 - angle1
                if dangle > np.pi:
                    dangle -= 2*np.pi
                elif dangle < -np.pi:
                    dangle += 2*np.pi
                    
                angular_velocity = dangle / dt
                angular_velocities.append(angular_velocity)
            
            # Calculate average values
            avg_speed = np.mean(velocities) if velocities else 0
            avg_angular_velocity = np.mean(angular_velocities) if angular_velocities else 0
            
            # Calculate radial distance statistics
            radial_distances = np.linalg.norm(positions[:, :2], axis=1)
            avg_radius = np.mean(radial_distances)
            max_radius = np.max(radial_distances)
            min_radius = np.min(radial_distances)
            
            # Calculate orbital period (if there is obvious periodic motion)
            if len(angular_velocities) > 0 and np.abs(avg_angular_velocity) > 1e-3:
                orbital_period = 2 * np.pi / np.abs(avg_angular_velocity)
            else:
                orbital_period = None
            
            # Calculate maximum speed and force
            v_max = np.sqrt(data['Vx (m/s)']**2 + data['Vy (m/s)']**2 + data['Vz (m/s)']**2).max()
            f_max = np.sqrt(data['Fx (N)']**2 + data['Fy (N)']**2 + data['Fz (N)']**2).max()
            
            if all(col in data.columns for col in ['τx (pN·μm)', 'τy (pN·μm)', 'τz (pN·μm)']):
                tau_max = np.sqrt(
                    data['τx (pN·μm)']**2 +
                    data['τy (pN·μm)']**2 +
                    data['τz (pN·μm)']**2
                ).max()
            else:
                tau_max = np.nan

            # Print statistics
            print(f"  Average speed: {avg_speed:.2e} m/s")
            print(f"  Maximum velocity: {v_max:.2e} m/s")
            print(f"  Average angular velocity: {avg_angular_velocity:.2e} rad/s")
            print(f"  Average radius: {avg_radius:.2e} m")
            print(f"  Maximum radius: {max_radius:.2e} m")
            print(f"  Minimum radius: {min_radius:.2e} m")
            if orbital_period is not None:
                print(f"  Orbital period: {orbital_period:.2e} s")
            else:
                print(f"  Orbital period: 无明显周期性运动 / No clear periodic motion")
            print(f"  Maximum force: {f_max:.2e} N")
            print(f"  Maximum torque: {tau_max:.2e} pN·μm")

            
            # Store statistics
            particle_stats = {
                'avg_speed': avg_speed,
                'avg_angular_velocity': avg_angular_velocity,
                'avg_radius': avg_radius,
                'max_radius': max_radius,
                'min_radius': min_radius,
                'orbital_period': orbital_period,
                'max_velocity': v_max,
                'max_force': f_max,
                'velocities': velocities,
                'angular_velocities': angular_velocities,
                'radial_distances': radial_distances
            }
            all_statistics[particle_id] = particle_stats
        
        return all_statistics

    def plot_2d_trajectory_with_phase(self, plane='xy', figsize=(12, 10), particle_ids=None, 
                                     optical_trap=None, field_alpha=0.6):
        """Plot 2D trajectory with optical field phase background
        
        Args:
            plane: Projection plane ('xy', 'xz', 'yz')
            figsize: Figure size
            particle_ids: List of particle IDs to plot, None for all particles
            optical_trap: OpticalTrap object for field phase
            field_alpha: Background field transparency (0-1)
        Returns:
            fig: Figure object
            ax: Axes object

        """
        if not self.particles_data:
            print("Please load data first")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot optical field phase background
        if optical_trap is not None and hasattr(optical_trap, 'phase') and optical_trap.phase is not None:
            self._plot_point_phase_background(ax, optical_trap, plane, field_alpha)
        
        # Determine particles to plot
        if particle_ids is None:
            particle_ids = list(self.particles_data.keys())
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(particle_ids)))
        
        for i, particle_id in enumerate(particle_ids):
            if particle_id not in self.particles_data:
                continue
                
            data = self.particles_data[particle_id]
            
            if plane == 'xy':
                ax.plot(data['X (m)'] * 1e6, data['Y (m)'] * 1e6, color=colors[i], 
                       linewidth=2, label=f'Particle {particle_id}', alpha=0.8)
                ax.scatter(data['X (m)'].iloc[0] * 1e6, data['Y (m)'].iloc[0] * 1e6, 
                          color=colors[i], s=100, marker='o', edgecolor='white', linewidth=2, zorder=10)
                ax.scatter(data['X (m)'].iloc[-1] * 1e6, data['Y (m)'].iloc[-1] * 1e6, 
                          color=colors[i], s=100, marker='s', edgecolor='white', linewidth=2, zorder=10)
                ax.set_xlabel('X (μm)')
                ax.set_ylabel('Y (μm)')
                ax.set_title('Particle Trajectory with Optical Field Phase (XY Plane)')
            elif plane == 'xz':
                ax.plot(data['X (m)'] * 1e6, data['Z (m)'] * 1e6, color=colors[i], 
                       linewidth=2, label=f'Particle {particle_id}', alpha=0.8)
                ax.scatter(data['X (m)'].iloc[0] * 1e6, data['Z (m)'].iloc[0] * 1e6, 
                          color=colors[i], s=100, marker='o', edgecolor='white', linewidth=2, zorder=10)
                ax.scatter(data['X (m)'].iloc[-1] * 1e6, data['Z (m)'].iloc[-1] * 1e6, 
                          color=colors[i], s=100, marker='s', edgecolor='white', linewidth=2, zorder=10)
                ax.set_xlabel('X (μm)')
                ax.set_ylabel('Z (μm)')
                ax.set_title('Particle Trajectory with Optical Field Phase (XZ Plane)')
            elif plane == 'yz':
                ax.plot(data['Y (m)'] * 1e6, data['Z (m)'] * 1e6, color=colors[i], 
                       linewidth=2, label=f'Particle {particle_id}', alpha=0.8)
                ax.scatter(data['Y (m)'].iloc[0] * 1e6, data['Z (m)'].iloc[0] * 1e6, 
                          color=colors[i], s=100, marker='o', edgecolor='white', linewidth=2, zorder=10)
                ax.scatter(data['Y (m)'].iloc[-1] * 1e6, data['Z (m)'].iloc[-1] * 1e6, 
                          color=colors[i], s=100, marker='s', edgecolor='white', linewidth=2, zorder=10)
                ax.set_xlabel('Y (μm)')
                ax.set_ylabel('Z (μm)')
                ax.set_title('Particle Trajectory with Optical Field Phase (YZ Plane)')
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        if optical_trap is not None and hasattr(optical_trap, 'phase') and optical_trap.phase is not None and hasattr(ax, 'images') and len(ax.images) > 0:
            cbar = plt.colorbar(ax.images[0], ax=ax, shrink=0.8)
            cbar.set_label('Phase (rad)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.show()

    def _plot_point_phase_background(self, ax, optical_trap, plane, alpha):
        """Plot point-wise optical field phase background"""
        grid_x = optical_trap.grid_x * 1e6
        grid_y = optical_trap.grid_y * 1e6
        grid_z = optical_trap.grid_z * 1e6
        phase = optical_trap.phase
        
        if plane == 'xy':
            z_center_idx = len(grid_z) // 2
            phase_slice = phase[:, :, z_center_idx]
            im = ax.imshow(phase_slice.T, extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                          origin='lower', cmap='hsv', alpha=alpha, aspect='equal', interpolation='bilinear')
        elif plane == 'xz':
            # Take slice at y=0 plane
            y_center_idx = len(grid_y) // 2
            phase_slice = phase[:, y_center_idx, :]
            X, Z = np.meshgrid(grid_x, grid_z, indexing='ij')
            
            im = ax.imshow(phase_slice.T, extent=[grid_x.min(), grid_x.max(), grid_z.min(), grid_z.max()],
                          origin='lower', cmap='hsv', alpha=alpha, aspect='equal', interpolation='bilinear')
        elif plane == 'yz':
            # Take slice at x=0 plane
            x_center_idx = len(grid_x) // 2
            phase_slice = phase[x_center_idx, :, :]
            Y, Z = np.meshgrid(grid_y, grid_z, indexing='ij')
            
            im = ax.imshow(phase_slice.T, extent=[grid_y.min(), grid_y.max(), grid_z.min(), grid_z.max()],
                          origin='lower', cmap='hsv', alpha=alpha, aspect='equal')
        
        return im

    def plot_2d_trajectory_with_field(self, plane='xy', figsize=(12, 10), particle_ids=None, 
                                     optical_trap=None, field_alpha=0.6, field_levels=20):
        """Plot 2D trajectory with optical field intensity background
        
        Args:
            plane: Projection plane ('xy', 'xz', 'yz')
            figsize: Figure size
            particle_ids: List of particle IDs to plot, None for all particles
            optical_trap: OpticalTrap object for field intensity
            field_alpha: Background field transparency (0-1)
            field_levels: Number of contour levels
        """
        if not self.particles_data:
            print("Please load data first")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot optical field intensity background
        if optical_trap is not None and optical_trap.field is not None:
            self._plot_field_background(ax, optical_trap, plane, field_alpha, field_levels)
        
        # Determine particles to plot
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
                # Mark start and end points
                ax.scatter(data['X (m)'].iloc[0] * 1e6, data['Y (m)'].iloc[0] * 1e6, 
                          color=colors[i], s=100, marker='o', edgecolor='white', linewidth=2, zorder=10)
                ax.scatter(data['X (m)'].iloc[-1] * 1e6, data['Y (m)'].iloc[-1] * 1e6, 
                          color=colors[i], s=100, marker='s', edgecolor='white', linewidth=2, zorder=10)
                ax.set_xlabel('X (μm)')
                ax.set_ylabel('Y (μm)')
                ax.set_title('Particle Trajectory with Optical Field Intensity (XY Plane)')
            elif plane == 'xz':
                ax.plot(data['X (m)'] * 1e6, data['Z (m)'] * 1e6, color=colors[i], 
                       linewidth=2, label=f'Particle {particle_id}', alpha=0.8)
                ax.scatter(data['X (m)'].iloc[0] * 1e6, data['Z (m)'].iloc[0] * 1e6, 
                          color=colors[i], s=100, marker='o', edgecolor='white', linewidth=2, zorder=10)
                ax.scatter(data['X (m)'].iloc[-1] * 1e6, data['Z (m)'].iloc[-1] * 1e6, 
                          color=colors[i], s=100, marker='s', edgecolor='white', linewidth=2, zorder=10)
                ax.set_xlabel('X (μm)')
                ax.set_ylabel('Z (μm)')
                ax.set_title('Particle Trajectory with Optical Field Intensity (XZ Plane)')
            elif plane == 'yz':
                ax.plot(data['Y (m)'] * 1e6, data['Z (m)'] * 1e6, color=colors[i], 
                       linewidth=2, label=f'Particle {particle_id}', alpha=0.8)
                ax.scatter(data['Y (m)'].iloc[0] * 1e6, data['Z (m)'].iloc[0] * 1e6, 
                          color=colors[i], s=100, marker='o', edgecolor='white', linewidth=2, zorder=10)
                ax.scatter(data['Y (m)'].iloc[-1] * 1e6, data['Z (m)'].iloc[-1] * 1e6, 
                          color=colors[i], s=100, marker='s', edgecolor='white', linewidth=2, zorder=10)
                ax.set_xlabel('Y (μm)')
                ax.set_ylabel('Z (μm)')
                ax.set_title('Particle Trajectory with Optical Field Intensity (YZ Plane)')
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        if optical_trap is not None and optical_trap.field is not None and hasattr(ax, 'collections') and len(ax.collections) > 0:
            cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.8)
            cbar.set_label('Normalized Intensity', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_field_background(self, ax, optical_trap, plane, alpha, levels):
        """Plot optical field intensity background"""
        # Get field grid and intensity data
        grid_x = optical_trap.grid_x * 1e6 
        grid_y = optical_trap.grid_y * 1e6
        grid_z = optical_trap.grid_z * 1e6
        field = optical_trap.field
        
        if plane == 'xy':
            # Take slice at z=0 plane
            z_center_idx = len(grid_z) // 2
            field_slice = field[:, :, z_center_idx]
            X, Y = np.meshgrid(grid_x, grid_y, indexing='ij')
            
            # Plot filled contours
            contour = ax.contourf(X, Y, field_slice, levels=levels, 
                                 cmap='hot', alpha=alpha, zorder=1)
            # Plot contour lines
            ax.contour(X, Y, field_slice, levels=levels, 
                      colors='white', alpha=0.3, linewidths=0.5, zorder=2)
            
        elif plane == 'xz':
            # Take slice at y=0 plane
            y_center_idx = len(grid_y) // 2
            field_slice = field[:, y_center_idx, :]
            X, Z = np.meshgrid(grid_x, grid_z, indexing='ij')
            
            contour = ax.contourf(X, Z, field_slice, levels=levels, 
                                 cmap='hot', alpha=alpha, zorder=1)
            ax.contour(X, Z, field_slice, levels=levels, 
                      colors='white', alpha=0.3, linewidths=0.5, zorder=2)
            
        elif plane == 'yz':
            # Take slice at x=0 plane
            x_center_idx = len(grid_x) // 2
            field_slice = field[x_center_idx, :, :]
            Y, Z = np.meshgrid(grid_y, grid_z, indexing='ij')
            
            contour = ax.contourf(Y, Z, field_slice, levels=levels, 
                                 cmap='hot', alpha=alpha, zorder=1)
            ax.contour(Y, Z, field_slice, levels=levels, 
                      colors='white', alpha=0.3, linewidths=0.5, zorder=2)
        
        return contour

    def plot_2d_trajectory_with_point_field(self, plane='xy', figsize=(12, 10), particle_ids=None,   # pyright: ignore[reportUnreachable]
                                           optical_trap=None, field_alpha=0.6):
        """Plot 2D trajectory with point-wise optical field intensity background
        
        Args:
            plane: Projection plane ('xy', 'xz', 'yz')
            figsize: Figure size
            particle_ids: List of particle IDs to plot, None for all particles
            optical_trap: OpticalTrap object for field intensity
            field_alpha: Background field transparency (0-1)
        """
        if not self.particles_data:
            print("Please load data first")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot optical field intensity background
        if optical_trap is not None and optical_trap.field is not None:
            self._plot_point_field_background(ax, optical_trap, plane, field_alpha)
        
        # Determine particles to plot
        if particle_ids is None:
            particle_ids = list(self.particles_data.keys())
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(particle_ids)))
        
        for i, particle_id in enumerate(particle_ids):
            if particle_id not in self.particles_data:
                continue
                
            data = self.particles_data[particle_id]
            
            if plane == 'xy':
                ax.plot(data['X (m)'] * 1e6, data['Y (m)'] * 1e6, color=colors[i], 
                       linewidth=2, label=f'Particle {particle_id}', alpha=0.8)
                # Mark start and end points
                ax.scatter(data['X (m)'].iloc[0] * 1e6, data['Y (m)'].iloc[0] * 1e6, 
                          color=colors[i], s=100, marker='o', edgecolor='white', linewidth=2, zorder=10)
                ax.scatter(data['X (m)'].iloc[-1] * 1e6, data['Y (m)'].iloc[-1] * 1e6, 
                          color=colors[i], s=100, marker='s', edgecolor='white', linewidth=2, zorder=10)
                ax.set_xlabel('X (μm)')
                ax.set_ylabel('Y (μm)')
                ax.set_title('Particle Trajectory with Point-wise Optical Field Intensity (XY Plane)')
            elif plane == 'xz':
                ax.plot(data['X (m)'] * 1e6, data['Z (m)'] * 1e6, color=colors[i], 
                       linewidth=2, label=f'Particle {particle_id}', alpha=0.8)
                ax.scatter(data['X (m)'].iloc[0] * 1e6, data['Z (m)'].iloc[0] * 1e6, 
                          color=colors[i], s=100, marker='o', edgecolor='white', linewidth=2, zorder=10)
                ax.scatter(data['X (m)'].iloc[-1] * 1e6, data['Z (m)'].iloc[-1] * 1e6, 
                          color=colors[i], s=100, marker='s', edgecolor='white', linewidth=2, zorder=10)
                ax.set_xlabel('X (μm)')
                ax.set_ylabel('Z (μm)')
                ax.set_title('Particle Trajectory with Point-wise Optical Field Intensity (XZ Plane)')
            elif plane == 'yz':
                ax.plot(data['Y (m)'] * 1e6, data['Z (m)'] * 1e6, color=colors[i], 
                       linewidth=2, label=f'Particle {particle_id}', alpha=0.8)
                ax.scatter(data['Y (m)'].iloc[0] * 1e6, data['Z (m)'].iloc[0] * 1e6, 
                          color=colors[i], s=100, marker='o', edgecolor='white', linewidth=2, zorder=10)
                ax.scatter(data['Y (m)'].iloc[-1] * 1e6, data['Z (m)'].iloc[-1] * 1e6, 
                          color=colors[i], s=100, marker='s', edgecolor='white', linewidth=2, zorder=10)
                ax.set_xlabel('Y (μm)')
                ax.set_ylabel('Z (μm)')
                ax.set_title('Particle Trajectory with Point-wise Optical Field Intensity (YZ Plane)')
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        if optical_trap is not None and optical_trap.field is not None and hasattr(ax, 'images') and len(ax.images) > 0:
            cbar = plt.colorbar(ax.images[0], ax=ax, shrink=0.8)
            cbar.set_label('Normalized Intensity', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_point_field_background(self, ax, optical_trap, plane, alpha):
        """Plot point-wise optical field intensity background"""

        grid_x = optical_trap.grid_x * 1e6 
        grid_y = optical_trap.grid_y * 1e6
        grid_z = optical_trap.grid_z * 1e6
        field = optical_trap.field
        
        if plane == 'xy':
            # Take slice at z=0 plane
            z_center_idx = len(grid_z) // 2
            field_slice = field[:, :, z_center_idx]
            
            im = ax.imshow(field_slice.T, extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                          origin='lower', cmap='hot', alpha=alpha, zorder=1, aspect='equal',
                          interpolation='bilinear')
            
        elif plane == 'xz':
            # Take slice at y=0 plane
            y_center_idx = len(grid_y) // 2
            field_slice = field[:, y_center_idx, :]
            
            im = ax.imshow(field_slice.T, extent=[grid_x.min(), grid_x.max(), grid_z.min(), grid_z.max()],
                          origin='lower', cmap='hot', alpha=alpha, zorder=1, aspect='equal',
                          interpolation='bilinear')
            
        elif plane == 'yz':
            # Take slice at x=0 plane
            x_center_idx = len(grid_x) // 2
            field_slice = field[x_center_idx, :, :]
            
            im = ax.imshow(field_slice.T, extent=[grid_y.min(), grid_y.max(), grid_z.min(), grid_z.max()],
                          origin='lower', cmap='hot', alpha=alpha, zorder=1, aspect='equal')
        
        return im

    def load_from_box(self, simulation_box, csv_file=None):
        """Load data and optical field from SimulationBox object
        
        Args:
            simulation_box: SimulationBox object
            csv_file: Optional CSV file path
        """
        try:
            if csv_file:
                self.load_data(csv_file)
            else:
                if hasattr(simulation_box, 'optical_trap') and simulation_box.optical_trap:
                    trap = simulation_box.optical_trap
                    if hasattr(trap, 'get_csv_paths'):
                        intensity_path, phase_path, success = trap.get_csv_paths()
                        if success:
                            print(f"Get the CSV path from optical_trap: {intensity_path}, {phase_path}")
                        else:
                            print("Cannot get CSV path from optical_trap")
                    else:
                        print("optical_trap does not have get_csv_paths method")
                else:
                    print("SimulationBox does not have optical_trap object")
            
            self.simulation_box = simulation_box
            print("Successfully load data from SimulationBox")
            
        except Exception as e:
            print(f"Failed to load data from SimulationBox: {e}")
    
    def load_from_trap(self, optical_trap, csv_file=None):
        """Load data and optical field from OpticalTrap object
        
        Args:
            optical_trap: OpticalTrap object
            csv_file: Optional CSV file path, if not provided, try to get from trap
        """
        try:
            if csv_file:
                self.load_data(csv_file)
            else:
                if hasattr(optical_trap, 'get_csv_paths'):
                    intensity_path, phase_path, success = optical_trap.get_csv_paths()
                    if success:
                        print(f"Get the CSV path from optical_trap: {intensity_path}, {phase_path}")
                    else:
                        print("Cannot get CSV path from optical_trap")
                else:
                    print("optical_trap does not have get_csv_paths method")
            
            self.optical_trap = optical_trap
            print("Successfully load data from OpticalTrap")
            
        except Exception as e:
            print(f"Failed to load data from OpticalTrap: {e}")

    def analyze_and_visualize_default(self, sim_box=None, show_plots=True):
        """Complete motion analysis and visualization with default configuration
        
        Args:
            sim_box: SimulationBox object, used to get optical trap information
            show_plots: Whether to display graphics, default is True
            
        Returns:
            dict: Statistical analysis results
        """
        try:
            print("\n=== Starting Analysis and Visualization with Default Configuration ===")
            

            if not self.particles_data:
                print("Please load data first")
                return None
            

            print("\n=== Motion Analysis using TrajectoryVisualizer ===")
            statistics = self.get_statistics()

            if statistics:
                print("\n=== Summary ===")
                print(f"Analysis completed for {len(statistics)} particle(s)")
                for particle_id, stats in statistics.items():
                    print(f"\nParticle {particle_id} key metrics:")
                    print(f"  Average speed: {stats['avg_speed']*1e6:.3f} μm/s")
                    print(f"  Average angular velocity: {stats['avg_angular_velocity']:.3f} rad/s")
                    print(f"  Average radius: {stats['avg_radius']*1e6:.2f} μm")
                    if stats['orbital_period'] is not None:
                        print(f"  Orbital period: {stats['orbital_period']:.3f} s")
                    else:
                        print("  Orbital period: Cannot calculate")

            
            print("\n=== Generating Visualizations ===")
            
            # 1. Velocity and angular velocity analysis
            print("Generating velocity and angular velocity analysis...")
            self.plot_velocity_magnitude()
            
            # 2. Force and torque analysis
            print("Generating force and torque analysis...")
            self.plot_force_magnitude()
            
            # 3. 3D trajectory
            print("Generating 3D trajectory...")
            self.plot_3d_trajectory()
            
            # 4. 2D trajectory with optical field background    
            if sim_box and hasattr(sim_box, 'optical_trap'):
                print("Generating trajectory with optical field background...")
                try:
                    self.plot_2d_trajectory_with_field(plane='xy', optical_trap=sim_box.optical_trap, field_alpha=0.6)
                    self.plot_2d_trajectory_with_phase(plane='xy', optical_trap=sim_box.optical_trap, field_alpha=0.6)
                except Exception as e:
                    print(f"Error when drawing optical field background: {e}")
            
            if show_plots:
                plt.show()
            
            print("\nAnalysis and visualization completed successfully!")
            
        except Exception as e:
            print(f"Analysis/Visualization error: {e}")
            return None
    

    def create_trajectory_video(self, output_filename='particle_trajectory_video.mp4', 
                            fps=30, duration_sec=10, trail_length=50, 
                            zoom_to_particle=True, optical_trap=None,
                            plane='xy', figsize=(12, 10)):
        """
        Create particle track video with optical field background

        
        Args:
            output_filename: Output video file name
            fps: Frame rate
            duration_sec: Video duration (seconds)
            trail_length: Trail length
            zoom_to_particle: Whether to zoom to particle motion range
            optical_trap: OpticalTrap object, used to display optical field background
            plane: Projection plane ('xy', 'xz', 'yz')
            figsize: Figure size
        """
        if not self.particles_data:
            print("Please load data first")
            return None
            
        
        print(f"Start generating video: {output_filename}")
        
        # Get the number and ID of particles
        particle_ids = list(self.particles_data.keys())
        num_particles = len(particle_ids)
        
        # Calculate the total number of frames
        total_frames = fps * duration_sec
        
        # Find the maximum number of data points
        max_data_points = max(len(self.particles_data[pid]) for pid in particle_ids)
        data_points_per_frame = max(1, max_data_points // total_frames)
        
        # Create the figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw the optical field background (if provided)
        if optical_trap is not None and hasattr(optical_trap, 'field') and optical_trap.field is not None:
            self._plot_field_background_for_video(ax, optical_trap, plane)
        
        # Set the axis labels
        if plane == 'xy':
            ax.set_xlabel('X Position (μm)', fontsize=12)
            ax.set_ylabel('Y Position (μm)', fontsize=12)
            title = f'Particle Motion in XY Plane ({num_particles} particles)'
        elif plane == 'xz':
            ax.set_xlabel('X Position (μm)', fontsize=12)
            ax.set_ylabel('Z Position (μm)', fontsize=12)
            title = f'Particle Motion in XZ Plane ({num_particles} particles)'
        elif plane == 'yz':
            ax.set_xlabel('Y Position (μm)', fontsize=12)
            ax.set_ylabel('Z Position (μm)', fontsize=12)
            title = f'Particle Motion in YZ Plane ({num_particles} particles)'
        
        ax.set_title(title, fontsize=14)
        
        # Define the colors
        colors = plt.cm.tab10(np.linspace(0, 1, min(num_particles, 10)))
        if num_particles > 10:
            colors = plt.cm.tab20(np.linspace(0, 1, min(num_particles, 20)))
        
        # Initialize the trail lines and particle points
        trail_lines = {}
        particle_points = {}
        
        for i, pid in enumerate(particle_ids):
            color = colors[i % len(colors)]
            trail_lines[pid], = ax.plot([], [], color=color, linewidth=2, alpha=0.8, 
                                    label=f'Particle {pid}')
            particle_points[pid], = ax.plot([], [], 'o', color=color, markersize=8, 
                                        markeredgecolor='white', markeredgewidth=2)
        
        # Add the time text
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add the particle count text
        count_text = ax.text(0.02, 0.02, f'Particles: {num_particles}', 
                            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Add the legend
        if num_particles <= 10:
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.legend(loc='upper right', fontsize=6, ncol=2)
        
        # 设置坐标轴范围
        all_data = pd.concat(self.particles_data.values())
        
        if plane == 'xy':
            x_col, y_col = 'X (m)', 'Y (m)'
        elif plane == 'xz':
            x_col, y_col = 'X (m)', 'Z (m)'
        elif plane == 'yz':
            x_col, y_col = 'Y (m)', 'Z (m)'
        
        if zoom_to_particle:
            x_margin = (all_data[x_col].max() - all_data[x_col].min()) * 1e6 * 0.2
            y_margin = (all_data[y_col].max() - all_data[y_col].min()) * 1e6 * 0.2
            
            x_range = [all_data[x_col].min() * 1e6 - x_margin, 
                    all_data[x_col].max() * 1e6 + x_margin]
            y_range = [all_data[y_col].min() * 1e6 - y_margin, 
                    all_data[y_col].max() * 1e6 + y_margin]
        else:

            if optical_trap is not None and hasattr(optical_trap, 'grid_x'):
                x_range = [optical_trap.grid_x.min() * 1e6, optical_trap.grid_x.max() * 1e6]
                y_range = [optical_trap.grid_y.min() * 1e6, optical_trap.grid_y.max() * 1e6]
            else:
                x_range = [-6, 6]
                y_range = [-6, 6]
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        def animate(frame):
            """Animation function for each frame"""
            current_idx = min(frame * data_points_per_frame, max_data_points - 1)
            
            # Update each particle
            for pid in particle_ids:
                particle_data = self.particles_data[pid]
                
                if current_idx < len(particle_data):
                    # Calculate the trail tail
                    trail_start = max(0, current_idx - trail_length)
                    
                    # Get the trail data
                    if plane == 'xy':
                        x_trail = particle_data[x_col].iloc[trail_start:current_idx+1] * 1e6
                        y_trail = particle_data[y_col].iloc[trail_start:current_idx+1] * 1e6
                        x_current = particle_data[x_col].iloc[current_idx] * 1e6
                        y_current = particle_data[y_col].iloc[current_idx] * 1e6
                    elif plane == 'xz':
                        x_trail = particle_data['X (m)'].iloc[trail_start:current_idx+1] * 1e6
                        y_trail = particle_data['Z (m)'].iloc[trail_start:current_idx+1] * 1e6
                        x_current = particle_data['X (m)'].iloc[current_idx] * 1e6
                        y_current = particle_data['Z (m)'].iloc[current_idx] * 1e6
                    elif plane == 'yz':
                        x_trail = particle_data['Y (m)'].iloc[trail_start:current_idx+1] * 1e6
                        y_trail = particle_data['Z (m)'].iloc[trail_start:current_idx+1] * 1e6
                        x_current = particle_data['Y (m)'].iloc[current_idx] * 1e6
                        y_current = particle_data['Z (m)'].iloc[current_idx] * 1e6
                    
                    # Update the trail line
                    trail_lines[pid].set_data(x_trail, y_trail)
                    
                    # Update the particle position
                    particle_points[pid].set_data([x_current], [y_current])
                    
                    # Update the time display (using the first particle's time)
                    if pid == particle_ids[0]:
                        current_time = particle_data['Time (s)'].iloc[current_idx]
                        time_text.set_text(f'Time: {current_time:.4f} s\nFrame: {frame+1}/{total_frames}')
                else:
                    # If no more data, hide the particle
                    trail_lines[pid].set_data([], [])
                    particle_points[pid].set_data([], [])
            
            return list(trail_lines.values()) + list(particle_points.values()) + [time_text, count_text]
        
        # Create animation
        print(f"Create animation, total frames: {total_frames}, data points per frame: {data_points_per_frame}")
        anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                    interval=1000//fps, blit=True, repeat=True)
        
        # Save the video
        print(f"Save video to: {output_filename}")  
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='ParticleTrajectoryVideo'), bitrate=1800)
            
            progress_bar = tqdm(total=total_frames, desc="Save video", unit="frame")
            
            def progress_callback(current_frame, total_frames):
                progress_bar.update(1)
            
            anim.save(output_filename, writer=writer, progress_callback=progress_callback)
            progress_bar.close()
            print(f"Video saved successfully: {output_filename}")
            
        except Exception as e:
            print(f"ffmpeg save failed: {e}")
            try:
                gif_filename = output_filename.replace('.mp4', '.gif')
                progress_bar = tqdm(total=total_frames, desc="Save GIF", unit="frame")
                
                def progress_callback(current_frame, total_frames):
                    progress_bar.update(1)
                
                anim.save(gif_filename, writer='pillow', fps=fps//2, progress_callback=progress_callback)
                progress_bar.close()
                print(f"GIF saved successfully: {gif_filename}")
            except Exception as e2:
                print(f"Save GIF failed: {e2}")
        
        plt.tight_layout()
        plt.show()
        
        return anim

    def _plot_field_background_for_video(self, ax, optical_trap, plane):
        """Drawing light field background for video"""
        if not hasattr(optical_trap, 'field') or optical_trap.field is None:
            return
        
        grid_x = optical_trap.grid_x * 1e6
        grid_y = optical_trap.grid_y * 1e6
        grid_z = optical_trap.grid_z * 1e6
        field = optical_trap.field
        
        if plane == 'xy':
            z_center_idx = len(grid_z) // 2
            field_slice = field[:, :, z_center_idx]
            im = ax.imshow(field_slice.T, 
                        extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                        origin='lower', cmap='hot', alpha=0.6,
                        norm=LogNorm(vmin=np.max(field_slice)*1e-6, vmax=np.max(field_slice)))
        elif plane == 'xz':
            y_center_idx = len(grid_y) // 2
            field_slice = field[:, y_center_idx, :]
            im = ax.imshow(field_slice.T, 
                        extent=[grid_x.min(), grid_x.max(), grid_z.min(), grid_z.max()],
                        origin='lower', cmap='hot', alpha=0.6,
                        norm=LogNorm(vmin=np.max(field_slice)*1e-6, vmax=np.max(field_slice)))
        elif plane == 'yz':
            x_center_idx = len(grid_x) // 2
            field_slice = field[x_center_idx, :, :]
            im = ax.imshow(field_slice.T, 
                        extent=[grid_y.min(), grid_y.max(), grid_z.min(), grid_z.max()],
                        origin='lower', cmap='hot', alpha=0.6,
                        norm=LogNorm(vmin=np.max(field_slice)*1e-6, vmax=np.max(field_slice)))
        
        # Add color bars
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Field Intensity (relative units)', fontsize=12)
        
        return im
