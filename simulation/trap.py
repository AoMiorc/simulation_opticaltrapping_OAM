import numpy as np

# Constants
from scipy.constants import c  
from scipy.constants import epsilon_0, mu_0 


class OpticalTrap:
    """Represents optical trap and its properties"""
    def __init__(self, center=None, wavelength=1064e-9, laser_power=0.1, w0=2e-6, l=np.nan):
        """Initialize optical trap
        
        Parameters:
        center (np.array): Trap center position [x, y, z] (m), defaults to origin
        wavelength (float): Laser wavelength (m), default 1064nm
        laser_power (float): Laser power (W), default 0.1W
        w0 (float): Beam waist radius (m), default 2 micrometers
        l (int): Orbital angular momentum quantum number, default np.nan(Indicates that the local l value is calculated from CSV phase data)
        """

        self.center = np.array([0.0, 0.0, 0.0]) if center is None else center
        self.field = None  # Optical field matrix (to be set in set_field method)
        self.wavelength = wavelength
        self.laser_power = laser_power
        self.w0 = w0
        self.l = l  # Orbital angular momentum quantum number
        self.grid_x = None
        self.grid_y = None
        self.grid_z = None
        
        # Initialize angular momentum related properties
        self.poynting_field = None
        self.angular_momentum_field = None
        self.axis_points = None  # Points on central axis
        self.axis_direction = None  # Direction vector of central axis
    
    def get_intensity_at_position(self, position):
        """Get normalized light intensity at specified position
        
        Parameters:
        position (np.array): Position vector [x, y, z] (m)
        
        Returns:
        float: Normalized light intensity at position
        """
        if self.field is None:
            return 0.0
            
        # Find nearest grid point
        x_idx = np.searchsorted(self.grid_x, position[0])
        y_idx = np.searchsorted(self.grid_y, position[1])
        z_idx = np.searchsorted(self.grid_z, position[2])
        
        # Ensure indices are within valid range
        x_idx = np.clip(x_idx, 0, len(self.grid_x)-1)
        y_idx = np.clip(y_idx, 0, len(self.grid_y)-1)
        z_idx = np.clip(z_idx, 0, len(self.grid_z)-1)
        
        return self.field[x_idx, y_idx, z_idx]

    def set_field(self, grid_x, grid_y, grid_z, field_function, phase_function=None):
        """Set optical field
        
        Parameters:
        grid_x (np.array): Grid points in x-direction
        grid_y (np.array): Grid points in y-direction
        grid_z (np.array): Grid points in z-direction
        field_function (function): Function to calculate field intensity
        phase_function (function, optional): Function to calculate phase
        """
        # Create grid
        X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
        
        # Calculate field intensity and phase
        self.field = field_function(X, Y, Z)
        self.phase = phase_function(X, Y, Z) if phase_function else np.zeros_like(self.field)
        
        # Store grid information
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
    
    def get_force(self, position, particle_radius=500e-9, refractive_index=1.5):
        """Calculate optical force based on intensity gradient
        
        Physical principle: F = -α ∇I(r)
        """
        if self.field is None:
            return np.zeros(3)
        
        # Particle polarization rate
        alpha = 4 * np.pi * epsilon_0 * particle_radius**3 * \
                (refractive_index**2 - 1) / (refractive_index**2 + 2)
        
        # Calculate intensity gradient
        gradient = self.calculate_intensity_gradient(position)
        if gradient is None:
            return np.zeros(3)
        
        # Calculate actual intensity (including power normalization)
        intensity_scale = self.laser_power / (np.pi * self.w0**2)  # W/m²
        
        # Optical force: F = α ∇I
        force = alpha * intensity_scale * gradient
        
        return force

    def calculate_intensity_gradient(self, position):
        """Calculate intensity gradient at specified position
        
        Parameters:
        position (np.array): Position vector [x, y, z] (m)
        
        Returns:
        np.array: Intensity gradient vector [∂I/∂x, ∂I/∂y, ∂I/∂z] or None if out of bounds
        """
        
        if self.grid_x is None or self.grid_y is None or self.grid_z is None:
            return None
        
        x, y, z = position
        
        # Check if position is within grid bounds
        if (x < self.grid_x[0] or x > self.grid_x[-1] or
            y < self.grid_y[0] or y > self.grid_y[-1] or
            z < self.grid_z[0] or z > self.grid_z[-1]):
            return np.zeros(3)  # Return zero gradient outside bounds
        
        # Find nearest grid point indices
        x_idx = np.searchsorted(self.grid_x, x)
        y_idx = np.searchsorted(self.grid_y, y)
        z_idx = np.searchsorted(self.grid_z, z)
        
        # Ensure indices are valid for gradient calculation
        x_idx = np.clip(x_idx, 1, len(self.grid_x)-2)
        y_idx = np.clip(y_idx, 1, len(self.grid_y)-2)
        z_idx = np.clip(z_idx, 1, len(self.grid_z)-2)
        
        # Calculate grid spacing
        dx = self.grid_x[1] - self.grid_x[0] 
        dy = self.grid_y[1] - self.grid_y[0]
        dz = self.grid_z[1] - self.grid_z[0]
        
        # Use central difference to calculate gradient
        dI_dx = (self.field[x_idx+1, y_idx, z_idx] - self.field[x_idx-1, y_idx, z_idx]) / (2 * dx)
        dI_dy = (self.field[x_idx, y_idx+1, z_idx] - self.field[x_idx, y_idx-1, z_idx]) / (2 * dy)
        dI_dz = (self.field[x_idx, y_idx, z_idx+1] - self.field[x_idx, y_idx, z_idx-1]) / (2 * dz)
        
        return np.array([dI_dx, dI_dy, dI_dz])



    def calculate_poynting_field(self):
        """Calculate Poynting vector based on local l value
        
        This is the optimized version of calculate_poynting_field_with_local_l,
        using vectorized operations for improved efficiency
        """
        if self.field is None or self.phase is None:
            print("Error: Optical field or phase field not initialized")
            return
        
        Z0 = np.sqrt(mu_0 / epsilon_0)
        X, Y, Z = np.meshgrid(self.grid_x, self.grid_y, self.grid_z, indexing='ij')
        
        # Calculate all positions' local l value field
        local_l_field = np.zeros_like(X)
        
        # Vectorized calculation of local l value
        if np.isnan(self.l) and self.phase is not None:
            dx = self.grid_x[1] - self.grid_x[0] if len(self.grid_x) > 1 else 1e-6
            dy = self.grid_y[1] - self.grid_y[0] if len(self.grid_y) > 1 else 1e-6
            
            # Calculate phase gradient
            dphase_dx, dphase_dy, _ = np.gradient(self.phase, dx, dy, 
                                            self.grid_z[1] - self.grid_z[0] if len(self.grid_z) > 1 else 1e-6)
            
            # Calculate radial distance and angle
            r = np.sqrt((X - self.center[0])**2 + (Y - self.center[1])**2)
            phi = np.arctan2(Y - self.center[1], X - self.center[0])
            
            # Calculate angular gradient    
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            
            # Avoid division by zero
            r_safe = np.where(r > 1e-10, r, 1e-10)
            angular_gradient = (sin_phi * dphase_dx - cos_phi * dphase_dy)
            
            # Calculate local l value
            local_l_field = r_safe * angular_gradient
            
            # In center region, use 0 as default
            center_mask = r < 1e-10
            local_l_field[center_mask] = 0
        elif not np.isnan(self.l):
            local_l_field.fill(self.l)
        else:
            local_l_field.fill(0)
        
        # Calculate coordinates
        r = np.sqrt((X - self.center[0])**2 + (Y - self.center[1])**2)
        phi = np.arctan2(Y - self.center[1], X - self.center[0])
        
        # Calculate electric field amplitude
        E0 = np.sqrt(2 * Z0 * self.field)
        
        # Calculate electric field components
        phase_with_oam = local_l_field * phi + self.phase
        
        # Electric field components (angularly polarized, adapted to local l value)
        Ex = -E0 * np.sin(phi) * np.exp(1j * phase_with_oam)
        Ey = E0 * np.cos(phi) * np.exp(1j * phase_with_oam)
        Ez = np.zeros_like(Ex)
        
        # Calculate magnetic field components (near-axis approximation)
        Hy = Ex / Z0
        Hx = -Ey / Z0
        Hz = np.zeros_like(Ex)
        
        # Calculate Poynting vector
        Sx = np.real(Ey * np.conj(Hz) - Ez * np.conj(Hy))
        Sy = np.real(Ez * np.conj(Hx) - Ex * np.conj(Hz))
        Sz = np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))
        
        self.poynting_field = np.stack([Sx, Sy, Sz], axis=-1)
        

        # 选择中心z平面进行积分
        z_center_idx = len(self.grid_z) // 2
        Sz_center = Sz[:, :, z_center_idx]
        total_power = np.sum(Sz_center) * (self.grid_x[1] - self.grid_x[0]) * \
                     (self.grid_y[1] - self.grid_y[0]) * (self.grid_z[1] - self.grid_z[0])
        if total_power > 0:
            self.poynting_field *= self.laser_power / total_power
        
        return self.poynting_field

    def get_poynting_vector_at_position(self, position):
        """Get Poynting vector at specified position
        
        Parameters:
        position (np.array): Position vector [x, y, z]
        
        Returns:
        np.array: Poynting vector [Sx, Sy, Sz]
        """
        if self.poynting_field is None:
            self.calculate_poynting_field()
            if self.poynting_field is None:
                return np.zeros(3)

        # Find nearest grid point coordinates
        x_idx = np.searchsorted(self.grid_x, position[0])
        y_idx = np.searchsorted(self.grid_y, position[1])
        z_idx = np.searchsorted(self.grid_z, position[2])

        # Ensure indices are within valid range
        x_idx = np.clip(x_idx, 0, len(self.grid_x)-1)
        y_idx = np.clip(y_idx, 0, len(self.grid_y)-1)
        z_idx = np.clip(z_idx, 0, len(self.grid_z)-1)

        return self.poynting_field[x_idx, y_idx, z_idx]

    def calculate_angular_momentum_field(self):
        """Calculate angular momentum field using Poynting vector
        L = r × S/c, where:
        - r is position vector
        - S is Poynting vector
        - c is speed of light
        """
        if self.poynting_field is None:
            self.calculate_poynting_field()
            if self.poynting_field is None:
                return None
    
        # Create grid points
        X, Y, Z = np.meshgrid(self.grid_x, self.grid_y, self.grid_z, indexing='ij')
    
        # Calculate position vector for each point (relative to beam center)
        R_x = X - self.center[0]
        R_y = Y - self.center[1]
        R_z = Z - self.center[2]
    
        # Get S components from poynting_field
        S_x = self.poynting_field[..., 0]
        S_y = self.poynting_field[..., 1]
        S_z = self.poynting_field[..., 2]
    
        # Calculate angular momentum density L = r × S/c
        L_x = (R_y * S_z - R_z * S_y) / c
        L_y = (R_z * S_x - R_x * S_z) / c
        L_z = (R_x * S_y - R_y * S_x) / c
    
        # Combine three components into vector field
        self.angular_momentum_field = np.stack([L_x, L_y, L_z], axis=-1)
    
        return self.angular_momentum_field
    
    def get_angular_momentum_at_position(self, position):
        """Get angular momentum density at specified position
        
        Parameters:
        position (np.array): Position vector [x, y, z]
        
        Returns:
        np.array: Angular momentum density [Lx, Ly, Lz]
        """
        if self.angular_momentum_field is None:
            self.calculate_angular_momentum_field()
            if self.angular_momentum_field is None:
                return np.zeros(3)
    
        # Find nearest grid point
        x_idx = np.searchsorted(self.grid_x, position[0])
        y_idx = np.searchsorted(self.grid_y, position[1])
        z_idx = np.searchsorted(self.grid_z, position[2])
    
        # Ensure indices are within valid range
        x_idx = np.clip(x_idx, 0, len(self.grid_x)-1)
        y_idx = np.clip(y_idx, 0, len(self.grid_y)-1)
        z_idx = np.clip(z_idx, 0, len(self.grid_z)-1)

        return self.angular_momentum_field[x_idx, y_idx, z_idx]

    def calculate_angular_momentum_axis(self):
        """Calculate central axis of angular momentum field
        
        Returns:
        np.array: Axis points [N, 3]
        np.array: Axis direction [3]
        """
        if self.angular_momentum_field is None:
            self.calculate_angular_momentum_field()
            if self.angular_momentum_field is None:
                self.axis_points = np.array([self.center])
                self.axis_direction = np.array([0, 0, 1])
                return self.axis_points, self.axis_direction
        
        try:
            nx, ny, nz = self.angular_momentum_field.shape[:3]
            axis_points = []
            
            for k in range(nz):
                L_z = self.angular_momentum_field[:, :, k, 2]
                
                positive_L_z = np.maximum(L_z, 0)
                total_weight = np.sum(positive_L_z)
                
                if total_weight > 0:
                    X_plane, Y_plane = np.meshgrid(self.grid_x, self.grid_y, indexing='ij')
                    
                    # Computational weighting center
                    center_x = np.sum(X_plane * positive_L_z) / total_weight
                    center_y = np.sum(Y_plane * positive_L_z) / total_weight
                    center_z = self.grid_z[k]
                    
                    axis_points.append([center_x, center_y, center_z])
                else:
                    # Use beam center if no positive z component
                    axis_points.append([self.center[0], self.center[1], self.grid_z[k]])
            
            axis_points = np.array(axis_points)
            
            # Improved axis direction calculation
            if len(axis_points) > 2:
                z_coords = axis_points[:, 2]
                x_coords = axis_points[:, 0]
                y_coords = axis_points[:, 1]
                
                #  Calculate gradient in z direction
                if len(set(z_coords)) > 1:  
                    dx_dz = np.polyfit(z_coords, x_coords, 1)[0]
                    dy_dz = np.polyfit(z_coords, y_coords, 1)[0]
                    axis_direction = np.array([dx_dz, dy_dz, 1])
                    axis_direction = axis_direction / np.linalg.norm(axis_direction)
                else:
                    axis_direction = np.array([0, 0, 1])
            else:
                axis_direction = np.array([0, 0, 1])
            
            self.axis_points = axis_points
            self.axis_direction = axis_direction
            return self.axis_points, self.axis_direction
            
        except Exception as e:
            print(f"Warning: Angular momentum axis calculation failed: {e}")
            self.axis_points = np.array([self.center])
            self.axis_direction = np.array([0, 0, 1])
            return self.axis_points, self.axis_direction

    def calculate_torque_at_position(self, position, particle, environment):
        """
        Calculate torque at given position using particle and environment properties
        
        Parameters:
        position (np.array): Position to calculate torque at
        particle (Particle): Particle object with refractive_index attribute
        environment (Environment): Environment object with refractive_index attribute
        
        Returns:
        np.array: Torque vector [Tx, Ty, Tz]
        """
        # Calculate relative refractive index
        m_rel = particle.refractive_index / environment.refractive_index
        
        # Calculate wave number
        k = 2 * np.pi / self.wavelength
        
        # Rayleigh scattering cross section
        sigma_sca = (8 * np.pi / 3) * (k ** 4) * (particle.radius ** 6) * abs((m_rel ** 2 - 1) / (m_rel ** 2 + 2)) ** 2

        # Get Poynting vector at position (W/m^2)
        S = self.get_poynting_vector_at_position(position)  # S = [Sx, Sy, Sz]

        # Calculate force: F = σ_sca * S / c
        force = sigma_sca * S / c  # Newtons

        # Position vector relative to beam center
        r_vec = np.array(position) - np.array(self.center)

        # Mechanical torque from force
        torque = np.cross(r_vec, force)  # N·m
        
        # Optical angular momentum per photon: l * ħ
        # ħ = reduced Planck constant
        hbar = 1.0545718e-34  # J·s

        # Photon energy E = h * c / λ
        E_photon = 6.62607015e-34 * c / self.wavelength

        # Get local orbital angular momentum quantum number l at position
        local_l = self.calculate_local_l_from_phase_gradient(position)

        # OAM torque: number of scattered photons per second * angular momentum per photon
        # Number of scattered photons per second = Power scattered / photon energy = σ_sca * (S · k̂) / E_photon
        # Here approximate k̂ as z direction unit vector
        S_dot_k = np.dot(S, np.array([0, 0, 1]))
        if S_dot_k < 0:
            S_dot_k = 0  # no negative power flow along beam direction

        power_scat = sigma_sca * S_dot_k  # Watts

        photon_rate = power_scat / E_photon  # photons per second

        # OAM torque magnitude = photon_rate * l * ħ
        torque_oam_z = photon_rate * local_l * hbar  # N·m

        torque[2] += torque_oam_z

        return torque



    def calculate_local_l_from_phase_gradient(self, position):
        """Calculate local effective l value from phase gradient"""
        if self.phase is None:
            return self.l 
        
        # Find the nearest grid point
        x_idx = np.searchsorted(self.grid_x, position[0])
        y_idx = np.searchsorted(self.grid_y, position[1])
        z_idx = np.searchsorted(self.grid_z, position[2])
        
        # Make sure the index is within the valid range
        x_idx = np.clip(x_idx, 1, len(self.grid_x)-2)
        y_idx = np.clip(y_idx, 1, len(self.grid_y)-2)
        z_idx = np.clip(z_idx, 0, len(self.grid_z)-1)
        
        # Calculate phase gradient
        dx = self.grid_x[1] - self.grid_x[0]
        dy = self.grid_y[1] - self.grid_y[0]
        
        phase_x_minus = self.phase[x_idx-1, y_idx, z_idx]
        phase_x_plus = self.phase[x_idx+1, y_idx, z_idx]
        phase_y_minus = self.phase[x_idx, y_idx-1, z_idx]
        phase_y_plus = self.phase[x_idx, y_idx+1, z_idx]

        phase_diff_x = phase_x_plus - phase_x_minus
        phase_diff_y = phase_y_plus - phase_y_minus


        phase_diff_x = np.arctan2(np.sin(phase_diff_x), np.cos(phase_diff_x))
        phase_diff_y = np.arctan2(np.sin(phase_diff_y), np.cos(phase_diff_y))

        # Calculate gradient
        dphase_dx = phase_diff_x / (2 * dx)
        dphase_dy = phase_diff_y / (2 * dy)
        # Calculate distance and angle to center
        r_vec = np.array(position) - np.array(self.center)
        r = np.sqrt(r_vec[0]**2 + r_vec[1]**2)
        
        if r > 1e-10:  
            # Calculate the angular gradient ∇φ_θ = sin(θ) * ∂φ/∂x - cos(θ) * ∂φ/∂y 
            cos_theta = r_vec[0] / r
            sin_theta = r_vec[1] / r
            
            angular_gradient = sin_theta * dphase_dx - cos_theta * dphase_dy
            
            # Local effective l value = r * ∇φ_θ
            local_l = r * angular_gradient
            
            return local_l
        else:
            return self.l 
    
    def load_csv_field_data(self, intensity_csv, phase_csv):
        """Loading light field intensity and phase data in CSV format
        
        Parameters:
        intensity_csv (str): Intensity data CSV file path
        phase_csv (str): Phase data CSV file path
        
        Returns:
        tuple: (intensity_data, phase_data, success) Returns data and True if successful, None, None, False if failed
        """
        try:
            
            # Load intensity and phase CSV data
            intensity_data = np.loadtxt(intensity_csv, delimiter=',')
            phase_data = np.loadtxt(phase_csv, delimiter=',')
            print(f"Successfully loaded intensity data, shape: {intensity_data.shape}")
            print(f"Successfully loaded phase data, shape: {phase_data.shape}")
            
            # Check if data shapes match
            if intensity_data.shape != phase_data.shape:
                print(f"Warning: Intensity and phase data shapes do not match!")
                return None, None, False
            
            return intensity_data, phase_data, True
            
        except Exception as e:
            print(f"Failed to load CSV files: {e}")
            return None, None, False

    def setup_csv_field(self, intensity_csv, phase_csv, x_range, y_range, z_range):
        """Setting up light field from CSV files - set_field method's CSV special alternative version
        
        This is an encapsulation function that specifically handles CSV format light field data. It can completely replace
        the traditional set_field + setup_csv_field_functions combination process.
        
        """
        try:
            from scipy.spatial import KDTree
            
            # Load CSV data
            intensity_data, phase_data, success = self.load_csv_field_data(intensity_csv, phase_csv)
            if not success:
                return False
                
            # Create target grid (these will become class attributes)
            if len(x_range) == 2: 
                grid_x = np.linspace(x_range[0], x_range[1], 100)
            else:
                grid_x = np.array(x_range)
                
            if len(y_range) == 2:
                grid_y = np.linspace(y_range[0], y_range[1], 100)
            else:
                grid_y = np.array(y_range)
                
            if len(z_range) == 2:
                grid_z = np.linspace(z_range[0], z_range[1], 50)
            else:
                grid_z = np.array(z_range)
            
            # Process CSV data format
            if len(intensity_data.shape) == 2:
                ny, nx = intensity_data.shape
                
                # Create CSV data corresponding coordinate grid
                x_csv = np.linspace(grid_x[0], grid_x[-1], nx)
                y_csv = np.linspace(grid_y[0], grid_y[-1], ny)
                X_csv, Y_csv = np.meshgrid(x_csv, y_csv)
                points = np.column_stack((X_csv.ravel(), Y_csv.ravel()))
                
                # Create KDTree
                tree = KDTree(points)
                
                # Define light field functions
                def csv_intensity_function(x, y, z):
                    """CSV data intensity interpolation function"""
                    points_query = np.column_stack((x.ravel(), y.ravel()))
                    dist, idx = tree.query(points_query)
                    result = intensity_data.ravel()[idx]
                    # Z-axis decay enhancement for physical realism
                    z_factor = np.exp(-(z.ravel()**2) / (2 * (1e-6)**2))
                    result *= z_factor
                    return result.reshape(x.shape)
                
                def csv_phase_function(x, y, z):
                    """CSV data phase interpolation function"""
                    points_query = np.column_stack((x.ravel(), y.ravel()))
                    dist, idx = tree.query(points_query)
                    result = phase_data.ravel()[idx]
                    return result.reshape(x.shape)
                
                self.set_field(grid_x, grid_y, grid_z, csv_intensity_function, csv_phase_function)
                
                # Verify grid attributes have been set correctly
                print(f"Grid setup successful:")
                print(f"  - grid_x: {len(self.grid_x)} points, range [{self.grid_x[0]:.2e}, {self.grid_x[-1]:.2e}]")
                print(f"  - grid_y: {len(self.grid_y)} points, range [{self.grid_y[0]:.2e}, {self.grid_y[-1]:.2e}]")
                print(f"  - grid_z: {len(self.grid_z)} points, range [{self.grid_z[0]:.2e}, {self.grid_z[-1]:.2e}]")
                
                return True
                
            else:
                print(f"Unsupported data format: {intensity_data.shape}")
                return False
                
        except Exception as e:
            print(f"Failed to setup CSV light field: {e}")
            return False

    def get_csv_paths(self, auto_generate=True, output_dir=None):
        """Get CSV file paths for intensity and phase data
        
        Parameters:
        auto_generate (bool): Whether to automatically generate CSV files if missing, default True
        output_dir (str): Directory to save generated CSV files, defaults to current directory
        
        Returns:
        tuple: (intensity_csv_path, phase_csv_path, success)
               Returns paths and True if successful, None, None, False if failed
        """
        import os
        
        # Check if both paths are set and files exist
        if (self.intensity_csv_path and self.phase_csv_path and 
            os.path.exists(self.intensity_csv_path) and os.path.exists(self.phase_csv_path)):
            return self.intensity_csv_path, self.phase_csv_path, True
        
        # If auto_generate is False and files don't exist, return failure
        if not auto_generate:
            return None, None, False
        
        # Generate CSV files if they don't exist
        if self.field is None or self.phase is None:
            print("Error: Optical field or phase field not initialized. Cannot generate CSV files.")
            return None, None, False
        
        try:
            # Set default output directory
            if output_dir is None:
                output_dir = os.getcwd()
            
            # Generate default filenames if not provided
            if not self.intensity_csv_path:
                self.intensity_csv_path = os.path.join(output_dir, "generated_intensity.csv")
            if not self.phase_csv_path:
                self.phase_csv_path = os.path.join(output_dir, "generated_phase.csv")
            
            # Generate intensity CSV file
            self._generate_intensity_csv(self.intensity_csv_path)
            
            # Generate phase CSV file  
            self._generate_phase_csv(self.phase_csv_path)
            
            print(f"Successfully generated CSV files:")
            print(f"  - Intensity: {self.intensity_csv_path}")
            print(f"  - Phase: {self.phase_csv_path}")
            
            return self.intensity_csv_path, self.phase_csv_path, True
            
        except Exception as e:
            print(f"Failed to generate CSV files: {e}")
            return None, None, False
    
    def _generate_intensity_csv(self, output_path):
        """Generate intensity CSV file from current field data
        
        Parameters:
        output_path (str): Output CSV file path
        """
        if self.field is None:
            raise ValueError("Field data not available")
        
        # Take a 2D slice at z=0 (center plane)
        z_center_idx = len(self.grid_z) // 2 if self.grid_z is not None else 0
        
        if len(self.field.shape) == 3:
            intensity_2d = self.field[:, :, z_center_idx]
        else:
            intensity_2d = self.field
        
        # Save to CSV
        np.savetxt(output_path, intensity_2d, delimiter=',')
        
    def _generate_phase_csv(self, output_path):
        """Generate phase CSV file from current phase data
        
        Parameters:
        output_path (str): Output CSV file path
        """
        if self.phase is None:
            raise ValueError("Phase data not available")
        
        # Take a 2D slice at z=0 (center plane)
        z_center_idx = len(self.grid_z) // 2 if self.grid_z is not None else 0
        
        if len(self.phase.shape) == 3:
            phase_2d = self.phase[:, :, z_center_idx]
        else:
            phase_2d = self.phase
        
        # Save to CSV
        np.savetxt(output_path, phase_2d, delimiter=',')
    
    def set_csv_paths(self, intensity_csv_path, phase_csv_path):
        """Set CSV file paths for intensity and phase data
        
        Parameters:
        intensity_csv_path (str): Path to intensity CSV file
        phase_csv_path (str): Path to phase CSV file
        """
        self.intensity_csv_path = intensity_csv_path
        self.phase_csv_path = phase_csv_path


