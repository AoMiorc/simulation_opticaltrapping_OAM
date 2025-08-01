import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import os
from scipy.interpolate import RegularGridInterpolator
import matplotlib
from tqdm import tqdm

# Set English fonts
matplotlib.rcParams.update({
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica'],
    'axes.unicode_minus': False,
    'font.size': 10,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 100,
    'savefig.dpi': 300
})

print(f"Current font setting: {matplotlib.rcParams['font.sans-serif']}")
print(f"Unicode minus setting: {matplotlib.rcParams['axes.unicode_minus']}")

class MultiParticleFieldVideoGenerator:
    def __init__(self, trajectory_csv, field_csv, field_size_cm=0.0012):
        """
        Initialize multi-particle video generator
        
        Args:
            trajectory_csv: Multi-particle trajectory CSV file path
            field_csv: Optical field intensity CSV file path
            field_size_cm: Field size in cm, default 0.0012cm = 12 micrometers
        """
        self.trajectory_csv = trajectory_csv
        self.field_csv = field_csv
        self.field_size_cm = field_size_cm
        self.field_size_m = field_size_cm / 100.0
        
        # Load data
        self.load_trajectory_data()
        self.load_field_data()
        
    def load_trajectory_data(self):
        """Load multi-particle trajectory data"""
        print(f"Loading trajectory data: {self.trajectory_csv}")
        self.trajectory_data = pd.read_csv(self.trajectory_csv)
        
        # Convert units to micrometers
        self.trajectory_data['X_um'] = self.trajectory_data['X (m)'] * 1e6
        self.trajectory_data['Y_um'] = self.trajectory_data['Y (m)'] * 1e6
        self.trajectory_data['Z_um'] = self.trajectory_data['Z (m)'] * 1e6
        
        # Get unique particle IDs
        self.particle_ids = sorted(self.trajectory_data['Particle_ID'].unique())
        self.num_particles = len(self.particle_ids)
        
        print(f"Trajectory data loaded, total {len(self.trajectory_data)} data points")
        print(f"Number of particles: {self.num_particles}")
        print(f"Particle IDs: {self.particle_ids}")
        
        # Print range information for all particles
        print(f"X range: {self.trajectory_data['X_um'].min():.3f} ~ {self.trajectory_data['X_um'].max():.3f} μm")
        print(f"Y range: {self.trajectory_data['Y_um'].min():.3f} ~ {self.trajectory_data['Y_um'].max():.3f} μm")
        print(f"Z range: {self.trajectory_data['Z_um'].min():.3f} ~ {self.trajectory_data['Z_um'].max():.3f} μm")
        
        # Separate data for each particle
        self.particle_data = {}
        for pid in self.particle_ids:
            particle_df = self.trajectory_data[self.trajectory_data['Particle_ID'] == pid].copy()
            particle_df = particle_df.sort_values('Time (s)').reset_index(drop=True)
            self.particle_data[pid] = particle_df
            print(f"Particle {pid}: {len(particle_df)} data points")
        
    def load_field_data(self):
        """Load optical field intensity data"""
        print(f"Loading field data: {self.field_csv}")
        
        try:
            # Load CSV data
            self.intensity_data = np.loadtxt(self.field_csv, delimiter=',')
            
            # Create coordinate grid (assuming square grid data)
            grid_size = int(np.sqrt(self.intensity_data.size))
            if grid_size * grid_size != self.intensity_data.size:
                self.intensity_data = self.intensity_data.reshape(-1, grid_size)
            else:
                self.intensity_data = self.intensity_data.reshape(grid_size, grid_size)
            
            # Create coordinate axes (consistent with trajectory generation)
            self.x_field = np.linspace(-6, 6, self.intensity_data.shape[1])  # micrometer units
            self.y_field = np.linspace(-6, 6, self.intensity_data.shape[0])  # micrometer units
            
            print(f"Field data loaded successfully, grid size: {self.intensity_data.shape}")
            print(f"Field range: ±6 μm")
            
        except Exception as e:
            print(f"Failed to load field data: {e}")
            print("Using default field")
            self.create_default_field()
    
    def create_default_field(self):
        """Create default LP71 mode optical field"""
        print("Creating default LP71 field...")
        
        x = np.linspace(-6, 6, 200)  # micrometer units
        y = np.linspace(-6, 6, 200)  # micrometer units
        X, Y = np.meshgrid(x, y)
        
        # LP71 mode approximate intensity distribution
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        
        radial_part = (r/3)**7 * np.exp(-(r/3)**2)
        angular_part = np.cos(theta)**2
        
        self.intensity_data = radial_part * angular_part
        self.x_field = x
        self.y_field = y
        
    def create_video(self, output_filename='multi_particle_field_video.mp4', 
                    fps=30, duration_sec=10, trail_length=50, zoom_to_particle=True):
        """
        Create multi-particle trajectory video with progress bar
        
        Args:
            output_filename: Output video filename
            fps: Frame rate
            duration_sec: Video duration (seconds)
            trail_length: Trajectory tail length
            zoom_to_particle: Whether to zoom to particle motion range
        """
        print(f"Starting multi-particle video generation: {output_filename}")
        
        # Calculate total frames
        total_frames = fps * duration_sec
        
        # Find the maximum number of data points among all particles
        max_data_points = max(len(self.particle_data[pid]) for pid in self.particle_ids)
        data_points_per_frame = max(1, max_data_points // total_frames)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Draw field background
        im = ax.imshow(self.intensity_data, 
                      extent=[self.x_field[0], self.x_field[-1], 
                             self.y_field[0], self.y_field[-1]],
                      origin='lower', 
                      cmap='hot', 
                      alpha=0.6,
                      norm=LogNorm(vmin=np.max(self.intensity_data)*1e-6, 
                                  vmax=np.max(self.intensity_data)))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Field Intensity (relative units)', fontsize=12)
        
        # Set axes
        ax.set_xlabel('X Position (μm)', fontsize=12)
        ax.set_ylabel('Y Position (μm)', fontsize=12)
        ax.set_title(f'Multi-Particle Motion in Optical Field ({self.num_particles} particles)', fontsize=14)
        
        # Define colors for different particles
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_particles))
        
        # Initialize trajectory lines and particle points for each particle
        trail_lines = {}
        particle_points = {}
        
        for i, pid in enumerate(self.particle_ids):
            color = colors[i]
            trail_lines[pid], = ax.plot([], [], color=color, linewidth=2, alpha=0.8, 
                                       label=f'Particle {pid}')
            particle_points[pid], = ax.plot([], [], 'o', color=color, markersize=8, 
                                           markeredgecolor='white', markeredgewidth=2)
        
        # Add time text
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add particle count text
        count_text = ax.text(0.02, 0.02, f'Particles: {self.num_particles}', 
                            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Add legend
        ax.legend(loc='upper right', fontsize=8, ncol=2 if self.num_particles > 5 else 1)
        
        # Set axis ranges
        if zoom_to_particle:
            x_margin = (self.trajectory_data['X_um'].max() - self.trajectory_data['X_um'].min()) * 0.2
            y_margin = (self.trajectory_data['Y_um'].max() - self.trajectory_data['Y_um'].min()) * 0.2
            
            x_range = [self.trajectory_data['X_um'].min() - x_margin, 
                      self.trajectory_data['X_um'].max() + x_margin]
            y_range = [self.trajectory_data['Y_um'].min() - y_margin, 
                      self.trajectory_data['Y_um'].max() + y_margin]
        else:
            x_range = [self.x_field[0], self.x_field[-1]]
            y_range = [self.y_field[0], self.y_field[-1]]
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        print(f"Display range: X=[{x_range[0]:.3f}, {x_range[1]:.3f}] μm, Y=[{y_range[0]:.3f}, {y_range[1]:.3f}] μm")
        
        def animate(frame):
            """Animation function for multi-particle"""
            current_idx = min(frame * data_points_per_frame, max_data_points - 1)
            
            # Update each particle
            for pid in self.particle_ids:
                particle_df = self.particle_data[pid]
                
                if current_idx < len(particle_df):
                    # Calculate trajectory tail
                    trail_start = max(0, current_idx - trail_length)
                    
                    # Get trajectory data
                    x_trail = particle_df['X_um'].iloc[trail_start:current_idx+1]
                    y_trail = particle_df['Y_um'].iloc[trail_start:current_idx+1]
                    
                    # Update trajectory line
                    trail_lines[pid].set_data(x_trail, y_trail)
                    
                    # Update particle position
                    x_current = particle_df['X_um'].iloc[current_idx]
                    y_current = particle_df['Y_um'].iloc[current_idx]
                    particle_points[pid].set_data([x_current], [y_current])
                    
                    # Update time display (use first particle's time)
                    if pid == self.particle_ids[0]:
                        current_time = particle_df['Time (s)'].iloc[current_idx]
                        time_text.set_text(f'Time: {current_time:.4f} s\nFrame: {frame+1}/{total_frames}')
                else:
                    # Hide particle if no more data
                    trail_lines[pid].set_data([], [])
                    particle_points[pid].set_data([], [])
            
            return list(trail_lines.values()) + list(particle_points.values()) + [time_text, count_text]
        
        # Create animation
        print(f"Creating animation, total frames: {total_frames}, data point interval: {data_points_per_frame}")
        anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                     interval=1000//fps, blit=True, repeat=True)
        
        # Save video with progress bar
        print(f"Saving video to: {output_filename}")
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='MultiParticleFieldVideo'), bitrate=1800)
            
            progress_bar = tqdm(total=total_frames, desc="Generating video", unit="frame")
            
            def progress_callback(current_frame, total_frames):
                progress_bar.update(1)
            
            anim.save(output_filename, writer=writer, progress_callback=progress_callback)
            progress_bar.close()
            print(f"Video saved successfully: {output_filename}")
            
        except Exception as e:
            print(f"Failed to save with ffmpeg: {e}")
            try:
                gif_filename = output_filename.replace('.mp4', '.gif')
                progress_bar = tqdm(total=total_frames, desc="Generating GIF", unit="frame")
                
                def progress_callback(current_frame, total_frames):
                    progress_bar.update(1)
                
                anim.save(gif_filename, writer='pillow', fps=fps//2, progress_callback=progress_callback)
                progress_bar.close()
                print(f"Saved as GIF format: {gif_filename}")
            except Exception as e2:
                print(f"Save failed: {e2}")
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def preview_field_and_trajectory(self):
        """Preview optical field and complete trajectories for all particles"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: field background + all trajectories
        im1 = ax1.imshow(self.intensity_data, 
                        extent=[self.x_field[0], self.x_field[-1], 
                               self.y_field[0], self.y_field[-1]],
                        origin='lower', cmap='hot', alpha=0.6)
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_particles))
        
        for i, pid in enumerate(self.particle_ids):
            particle_df = self.particle_data[pid]
            color = colors[i]
            
            ax1.plot(particle_df['X_um'], particle_df['Y_um'], 
                    color=color, linewidth=2, alpha=0.8, label=f'Particle {pid}')
            ax1.plot(particle_df['X_um'].iloc[0], particle_df['Y_um'].iloc[0], 
                    'o', color=color, markersize=6, markeredgecolor='white')
            ax1.plot(particle_df['X_um'].iloc[-1], particle_df['Y_um'].iloc[-1], 
                    's', color=color, markersize=6, markeredgecolor='white')
        
        ax1.set_xlabel('X Position (μm)')
        ax1.set_ylabel('Y Position (μm)')
        ax1.set_title(f'Field Background + All Trajectories ({self.num_particles} particles)')
        ax1.legend(fontsize=8, ncol=2 if self.num_particles > 5 else 1)
        ax1.grid(True, alpha=0.3)
        
        # Right plot: trajectory details with time coloring
        for i, pid in enumerate(self.particle_ids):
            particle_df = self.particle_data[pid]
            
            scatter = ax2.scatter(particle_df['X_um'], particle_df['Y_um'], 
                                 c=particle_df['Time (s)'], cmap='viridis', 
                                 s=20, alpha=0.6, label=f'Particle {pid}')
        
        plt.colorbar(scatter, ax=ax2, label='Time (s)')
        
        ax2.set_xlabel('X Position (μm)')
        ax2.set_ylabel('Y Position (μm)')
        ax2.set_title('Trajectory Details (Colored by Time)')
        ax2.legend(fontsize=8, ncol=2 if self.num_particles > 5 else 1)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('multi_particle_field_trajectory_preview.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # File paths - adjust according to your ten-particle simulation output
    trajectory_file = 'particle_trajectory_lp71_csv_ten_particles.csv'
    field_file = 'test0714/final_intensity_LP71_m6_2cm.csv'  # Use same field file as simulation
    
    # Check if files exist
    if not os.path.exists(trajectory_file):
        print(f"Error: Cannot find trajectory file {trajectory_file}")
        print("Please run the ten-particle simulation first:")
        print("python test0714/test_lp71_csv_no_interp_ten_particles.py")
        return
    
    if not os.path.exists(field_file):
        print(f"Warning: Cannot find field file {field_file}, will use default field")
    
    # Create multi-particle video generator
    video_gen = MultiParticleFieldVideoGenerator(trajectory_file, field_file, field_size_cm=0.0012)
    
    # Generate preview image
    print("Generating preview image...")
    video_gen.preview_field_and_trajectory()
    
    # Generate zoomed video focusing on particle motion
    print("\nGenerating zoomed video (focusing on particle motion)...")
    anim1 = video_gen.create_video('multi_particle_field_video_zoomed.mp4', 
                                  fps=30, duration_sec=15, trail_length=100, 
                                  zoom_to_particle=True)
    
    # Generate full view video
    print("\nGenerating full view video...")
    anim2 = video_gen.create_video('multi_particle_field_video_full.mp4', 
                                  fps=30, duration_sec=15, trail_length=100, 
                                  zoom_to_particle=False)
    
    print("\nMulti-particle video generation completed!")
    print("Generated files:")
    print("- multi_particle_field_trajectory_preview.png (preview image)")
    print("- multi_particle_field_video_zoomed.mp4 (zoomed video, focusing on particle motion)")
    print("- multi_particle_field_video_full.mp4 (full view video)")

if __name__ == "__main__":
    main()