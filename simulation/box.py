import numpy as np
from scipy.constants import Boltzmann as k_B

class SimulationBox:
    """3D simulation canvas that integrates particles, optical traps and environment"""
    
    def __init__(self, particles=None, environment=None, optical_trap=None, timestep=1e-6):
        """Initialize simulation box
        
        Parameters:
            particles: Single particle object or particle object list
            environment: Environment Objects
            optical_trap: Optical Trap Objects
            timestep: Time step (s), default 1μs
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
            
    
    def _step(self, particle_index=None):
        """Update particle state (internal method)
        
        Parameters:
            particle_index: Index of particle to update, if None update all particles
        """
        if particle_index is None:
            particles_to_update = self.particles
        else:
            if 0 <= particle_index < len(self.particles):
                particles_to_update = [self.particles[particle_index]]
            else:
                print(f"Warning: Particle index {particle_index} out of range, updating all particles")
                particles_to_update = self.particles
        
        for particle in particles_to_update:
            # Calculate optical force
            optical_force = self.optical_trap.get_force(particle.position)
            
            # Calculate current particle's damping coefficient
            gamma = self.environment.get_drag_coefficient(particle)
            
            # Calculate random fluctuation force
            variance = 2 * gamma * k_B * self.environment.T / self.timestep
            fluctuation_force = np.random.normal(0, np.sqrt(variance), 3)
            
            # v_{n+1} = (v_n + (F_optical + F_random) * dt / m) / (1 + γ * dt / m)
            non_damping_force = optical_force + fluctuation_force
            
            # Update velocity (semi-implicit method)
            damping_factor = 1 + gamma * self.timestep / particle.mass
            velocity_increment = non_damping_force * self.timestep / particle.mass
            particle.velocity = (particle.velocity + velocity_increment) / damping_factor
            

            # Calculate optical torque
            optical_torque = self.optical_trap.calculate_torque_at_position(particle.position, particle, self.environment)
            torque_z = optical_torque[2] 
            
            # Determine the rotation center
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
            
            # Update position
            particle.position += particle.velocity * self.timestep
            
            # Update total force
            total_force = optical_force + fluctuation_force
            
            particle.acceleration = total_force / particle.mass
            
            # Update time
            self.time += self.timestep
            
            # Record trajectory and state
            particle_idx = self.particles.index(particle)
            self.trajectory[particle_idx].append((self.time, particle.position.copy()))
            self.velocity_history[particle_idx].append(particle.velocity.copy())
            self.force_history[particle_idx].append(total_force.copy())
            self.angular_trajectory[particle_idx].append((self.time, particle.angular_velocity.copy()))
            self.torque_history[particle_idx].append(optical_torque.copy())
        
        # Return updated particle positions
        if particle_index is not None and 0 <= particle_index < len(self.particles):
            return self.particles[particle_index].position
        else:
            return [particle.position for particle in self.particles]
    
    def simulate(self, duration, show_progress=True):
        """Run simulation
        
        Parameters:
            duration: Simulation duration
            save_interval: Save interval
            show_progress: Whether to show progress bar
        """
        num_steps = int(duration / self.timestep)
        self.trajectory = [[] for _ in self.particles]
        self.velocity_history = [[] for _ in self.particles]
        self.force_history = [[] for _ in self.particles]
        self.angular_trajectory = [[] for _ in self.particles]
        self.torque_history = [[] for _ in self.particles]
        
        # Progress bar setup
        if show_progress:
            print(f"Running simulation: {num_steps} steps, timestep: {self.timestep:.2e}s")
            print(f"Expected duration: {duration}s")
            progress_interval = max(1, num_steps // 100)
        
        for step in range(num_steps):
            self._step()
            
            # Show progress
            if show_progress and (step + 1) % progress_interval == 0:
                progress = (step + 1) / num_steps * 100
                bar_length = 50
                filled_length = int(bar_length * (step + 1) // num_steps)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f'\rProgress: |{bar}| {progress:.1f}% ({step + 1}/{num_steps} steps)', end='', flush=True)
        
        if show_progress:
            print('\nSimulation completed!')
        
        return self.get_trajectory()
    
    
    def get_trajectory(self):
        """Get trajectory data for all particles"""
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
        """ Save multi-particle trajectory data to CSV file"""
        trajectories = self.get_trajectory() 
        
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            # Write to the header
            f.write("Particle_ID,Time (s),X (m),Y (m),Z (m),Vx (m/s),Vy (m/s),Vz (m/s),Fx (N),Fy (N),Fz (N),ωx (rad/s),ωy (rad/s),ωz (rad/s),τx (pN·μm),τy (pN·μm),τz (pN·μm)\n")
            

            for particle_id, data in enumerate(trajectories):
                for i in range(len(data['time'])):
                    t = data['time'][i]
                    x, y, z = data['position'][i]
                    vx, vy, vz = data['velocity'][i]
                    fx, fy, fz = data['force'][i]
                    ωx, ωy, ωz = data['angular_velocity'][i]
                    τx, τy, τz = data['torque'][i]
                    
                    # Convert torque unit: N⋅m → pN⋅μm (multiply by 10^18)
                    τx_pN_um = τx * 1e18
                    τy_pN_um = τy * 1e18
                    τz_pN_um = τz * 1e18
                    
                    f.write(f"{particle_id},{t:.6e},{x:.6e},{y:.6e},{z:.6e},{vx:.6e},{vy:.6e},{vz:.6e},{fx:.6e},{fy:.6e},{fz:.6e},{ωx:.6e},{ωy:.6e},{ωz:.6e},{τx_pN_um:.6e},{τy_pN_um:.6e},{τz_pN_um:.6e}\n")