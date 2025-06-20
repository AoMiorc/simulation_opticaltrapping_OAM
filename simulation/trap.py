import numpy as np

# 常量 / Constants
from scipy.constants import c  # 光速 / Speed of light
from scipy.constants import epsilon_0, mu_0  # 真空介电常数和磁导率 / Vacuum permittivity and permeability
from scipy.constants import pi  


class OpticalTrap:
    """表示光阱及其属性 / Represents optical trap and its properties"""
    def __init__(self, kappa, center=None, wavelength=1064e-9, laser_power=0.1, w0=2e-6, l=1):
        """初始化光阱 / Initialize optical trap
        
        参数 / Parameters:
        kappa (np.array): 三个方向的阱刚度 [κ_x, κ_y, κ_z] (N/m) / Trap stiffness in three directions [κ_x, κ_y, κ_z] (N/m)
        center (np.array): 阱中心位置 [x, y, z] (m)，默认为原点 / Trap center position [x, y, z] (m), defaults to origin
        wavelength (float): 激光波长 (m)，默认1064nm / Laser wavelength (m), default 1064nm
        laser_power (float): 激光功率 (W)，默认0.1W / Laser power (W), default 0.1W
        w0 (float): 束腰半径 (m)，默认2微米 / Beam waist radius (m), default 2 micrometers
        l (int): 轨道角动量量子数，默认1 / Orbital angular momentum quantum number, default 1
        """
        self.kappa = np.array(kappa)  # 阱刚度 / Trap stiffness
        self.center = np.array([0.0, 0.0, 0.0]) if center is None else center
        self.field = None  # 光场矩阵（将在set_field方法中设置） / Optical field matrix (to be set in set_field method)
        self.wavelength = wavelength
        self.laser_power = laser_power
        self.w0 = w0
        self.l = l  # 轨道角动量量子数 / Orbital angular momentum quantum number
        self.grid_x = None
        self.grid_y = None
        self.grid_z = None
        
        # 初始化角动量相关属性 / Initialize angular momentum related properties
        self.poynting_field = None
        self.angular_momentum_field = None
        self.axis_points = None  # 中心轴上的点 / Points on central axis
        self.axis_direction = None  # 中心轴的方向向量 / Direction vector of central axis
    
    def get_intensity_at_position(self, position):
        """获取指定位置的归一化光强 / Get normalized light intensity at specified position"""
        if self.field is None:
            return 0.0
            
        # 找到最近的网格点 / Find nearest grid point
        x_idx = np.searchsorted(self.grid_x, position[0])
        y_idx = np.searchsorted(self.grid_y, position[1])
        z_idx = np.searchsorted(self.grid_z, position[2])
        
        # 确保索引在有效范围内 / Ensure indices are within valid range
        x_idx = np.clip(x_idx, 0, len(self.grid_x)-1)
        y_idx = np.clip(y_idx, 0, len(self.grid_y)-1)
        z_idx = np.clip(z_idx, 0, len(self.grid_z)-1)
        
        return self.field[x_idx, y_idx, z_idx]

    def set_field(self, grid_x, grid_y, grid_z, field_function, phase_function=None):
        """设置光场 / Set optical field"""
        # 创建网格 / Create grid
        X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
        
        # 计算场强度和相位 / Calculate field intensity and phase
        self.field = field_function(X, Y, Z)
        self.phase = phase_function(X, Y, Z) if phase_function else np.zeros_like(self.field)
        
        # 存储网格信息 / Store grid information
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
    
    def get_force(self, position):
        """计算光学力 / Calculate optical force
        F = κ * x * direction，其中κ是阱刚度，x是到最近峰值的距离，direction是指向峰值的单位向量
        F = κ * x * direction, where κ is trap stiffness, x is distance to nearest peak, direction is unit vector pointing to peak
        """
        
        if self.field is None:
            return np.zeros(3)
        
        # 找到光场中的所有峰值点 / Find all peak points in optical field
        peak_positions = self.find_field_peaks()
        
        if len(peak_positions) == 0:
            # 如果没有找到峰值，返回零力 / If no peaks found, return zero force
            return np.zeros(3)
        
        # 找到距离当前位置最近的峰值 / Find nearest peak to current position
        distances = []
        for peak_pos in peak_positions:
            dist = np.linalg.norm(np.array(peak_pos) - np.array(position))
            distances.append(dist)
        
        nearest_peak_idx = np.argmin(distances)
        nearest_peak_pos = peak_positions[nearest_peak_idx]
        distance_to_peak = distances[nearest_peak_idx]
        
        # 计算指向峰值的方向向量 / Calculate direction vector pointing to peak
        direction_vector = np.array(nearest_peak_pos) - np.array(position)
        
        # 如果距离为零，返回零力 / If distance is zero, return zero force
        if distance_to_peak == 0:
            return np.zeros(3)
        
        # 归一化方向向量 / Normalize direction vector
        direction_unit = direction_vector / distance_to_peak
        
        # 计算光学力：F = κ * x * direction / Calculate optical force: F = κ * x * direction
        # 使用三个方向的平均kappa值 / Use average kappa value from three directions
        kappa_avg = np.mean(self.kappa)
        force_magnitude = kappa_avg * distance_to_peak
        force = force_magnitude * direction_unit
        
        return force 
    
    def find_field_peaks(self):
        """找到光场中的峰值点 / Find peak points in optical field
        
        返回 / Returns:
        list: 峰值点的坐标列表，每个元素为 [x, y, z] / List of peak point coordinates, each element is [x, y, z]
        """
        if self.field is None:
            return []
        
        # 检查field是否包含有效数值 / Check if field contains valid values
        if not np.isfinite(self.field).any():
            print("警告：光场包含无效数值，使用中心点作为峰值 / Warning: Optical field contains invalid values, using center point as peak")
            return [[self.center[0], self.center[1], self.center[2]]]
        
        from scipy.ndimage import maximum_filter
        from scipy.ndimage import label
        
        # 使用局部最大值滤波器找到峰值 / Use local maximum filter to find peaks
        # 定义邻域大小（3x3x3） / Define neighborhood size (3x3x3)
        neighborhood_size = 3
        local_maxima = maximum_filter(self.field, size=neighborhood_size) == self.field
        
        # 设置阈值，只考虑强度较高的峰值 / Set threshold, only consider high intensity peaks
        threshold = 0.1 * np.max(self.field)  # 阈值设为最大强度的10% / Threshold set to 10% of maximum intensity
        significant_peaks = local_maxima & (self.field > threshold)
        
        # 获取峰值点的索引 / Get indices of peak points
        peak_indices = np.where(significant_peaks)
        
        # 转换为实际坐标 / Convert to actual coordinates
        peak_positions = []
        for i in range(len(peak_indices[0])):
            x_idx = peak_indices[0][i]
            y_idx = peak_indices[1][i]
            z_idx = peak_indices[2][i]
            
            # 检查索引是否有效 / Check if indices are valid
            if (x_idx < len(self.grid_x) and y_idx < len(self.grid_y) and z_idx < len(self.grid_z)):
                x_coord = self.grid_x[x_idx]
                y_coord = self.grid_y[y_idx]
                z_coord = self.grid_z[z_idx]
                
                # 确保坐标是有限数值 / Ensure coordinates are finite values
                if np.isfinite([x_coord, y_coord, z_coord]).all():
                    peak_positions.append([x_coord, y_coord, z_coord])
        
        # 如果没有找到有效峰值，使用中心点 / If no valid peaks found, use center point
        if len(peak_positions) == 0:
            peak_positions = [[self.center[0], self.center[1], self.center[2]]]
        
        return peak_positions

    def calculate_poynting_field(self):
        """基于相位场计算坡印廷矢量 / Calculate Poynting vector based on phase field"""
        if self.field is None or self.phase is None:
            print("错误：光场或相位场未初始化，请先调用 set_field 方法设置光场和相位 / Error: Optical field or phase field not initialized, please call set_field method first")
            return
        
        # 计算真空阻抗 / Calculate vacuum impedance
        Z0 = np.sqrt(mu_0 / epsilon_0)
        
        # 计算电场振幅 / Calculate electric field amplitude
        E_amp = np.sqrt(2 * Z0 * self.field)
        
        # 使用复数表示电场 / Use complex representation for electric field
        E_complex = E_amp * np.exp(1j * self.phase)
        
        # 假设电场主要在x方向，y方向分量由相位梯度决定 / Assume electric field mainly in x direction, y component determined by phase gradient
        Ex = E_complex
        
        # 计算y方向电场分量（与相位梯度相关） / Calculate y-direction electric field component (related to phase gradient)
        # 使用中心差分计算相位梯度 / Use central difference to calculate phase gradient
        grad_phase_x = np.gradient(self.phase, axis=0, edge_order=2)
        grad_phase_y = np.gradient(self.phase, axis=1, edge_order=2)
        
        # y方向电场分量与相位梯度成正比 / y-direction electric field component proportional to phase gradient
        Ey = E_complex * (grad_phase_y + 1j * grad_phase_x)
        
        # z方向电场分量为0（假设主要在xy平面） / z-direction electric field component is 0 (assuming mainly in xy plane)
        Ez = np.zeros_like(Ex)
        
        # 计算磁场（与电场垂直且满足真空中的关系） / Calculate magnetic field (perpendicular to electric field and satisfying vacuum relations)
        Hx = -Ey / Z0
        Hy = Ex / Z0
        Hz = np.zeros_like(Ex)
        
        # 计算坡印廷矢量（取实部，因为我们关心的是时间平均的能流） / Calculate Poynting vector (take real part as we care about time-averaged energy flow)
        Sx = np.real(Ey * np.conj(Hz) - Ez * np.conj(Hy))
        Sy = np.real(Ez * np.conj(Hx) - Ex * np.conj(Hz))
        Sz = np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))
        
        self.poynting_field = np.stack([Sx, Sy, Sz], axis=-1)
        
        # 归一化到设定的激光功率 / Normalize to set laser power
        S_magnitude = np.sqrt(Sx**2 + Sy**2 + Sz**2)
        total_power = np.sum(S_magnitude) * (self.grid_x[1] - self.grid_x[0]) * \
                          (self.grid_y[1] - self.grid_y[0]) * (self.grid_z[1] - self.grid_z[0])
        self.poynting_field *= self.laser_power / total_power
    
        return self.poynting_field

    def get_poynting_vector_at_position(self, position):
        """获取指定位置的Poynting矢量 / Get Poynting vector at specified position"""
        if self.poynting_field is None:
            self.calculate_poynting_field()
            if self.poynting_field is None:
                return np.zeros(3)

        # 找到最近的网格点 / Find nearest grid point
        x_idx = np.searchsorted(self.grid_x, position[0])
        y_idx = np.searchsorted(self.grid_y, position[1])
        z_idx = np.searchsorted(self.grid_z, position[2])

        # 确保索引在有效范围内 / Ensure indices are within valid range
        x_idx = np.clip(x_idx, 0, len(self.grid_x)-1)
        y_idx = np.clip(y_idx, 0, len(self.grid_y)-1)
        z_idx = np.clip(z_idx, 0, len(self.grid_z)-1)

        return self.poynting_field[x_idx, y_idx, z_idx]

    def calculate_angular_momentum_field(self):
        """使用坡印廷矢量计算角动量场 / Calculate angular momentum field using Poynting vector
        L = r × S/c，其中： / L = r × S/c, where:
        - r 是位置矢量 / r is position vector
        - S 是坡印廷矢量 / S is Poynting vector
        - c 是光速 / c is speed of light
        """
        if self.poynting_field is None:
            self.calculate_poynting_field()
            if self.poynting_field is None:
                return None
    
        # 创建网格点 / Create grid points
        X, Y, Z = np.meshgrid(self.grid_x, self.grid_y, self.grid_z, indexing='ij')
    
        # 计算每个点的位置矢量（相对于光束中心） / Calculate position vector for each point (relative to beam center)
        R_x = X - self.center[0]
        R_y = Y - self.center[1]
        R_z = Z - self.center[2]
    
        # 从poynting_field获取S的分量 / Get S components from poynting_field
        S_x = self.poynting_field[..., 0]
        S_y = self.poynting_field[..., 1]
        S_z = self.poynting_field[..., 2]
    
        # 计算角动量密度 L = r × S/c / Calculate angular momentum density L = r × S/c
        L_x = (R_y * S_z - R_z * S_y) / c
        L_y = (R_z * S_x - R_x * S_z) / c
        L_z = (R_x * S_y - R_y * S_x) / c
    
        # 将三个分量组合成矢量场 / Combine three components into vector field
        self.angular_momentum_field = np.stack([L_x, L_y, L_z], axis=-1)
    
        return self.angular_momentum_field
    
    def get_angular_momentum_at_position(self, position):
        """获取指定位置的角动量密度 / Get angular momentum density at specified position"""
        if self.angular_momentum_field is None:
            self.calculate_angular_momentum_field()
            if self.angular_momentum_field is None:
                return np.zeros(3)
    
        # 找到最近的网格点 / Find nearest grid point
        x_idx = np.searchsorted(self.grid_x, position[0])
        y_idx = np.searchsorted(self.grid_y, position[1])
        z_idx = np.searchsorted(self.grid_z, position[2])
    
        # 确保索引在有效范围内 / Ensure indices are within valid range
        x_idx = np.clip(x_idx, 0, len(self.grid_x)-1)
        y_idx = np.clip(y_idx, 0, len(self.grid_y)-1)
        z_idx = np.clip(z_idx, 0, len(self.grid_z)-1)

        return self.angular_momentum_field[x_idx, y_idx, z_idx]

    def calculate_angular_momentum_axis(self):
        """计算角动量场的中心轴 / Calculate central axis of angular momentum field"""
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
                # 使用角动量的z分量作为权重（更物理合理）
                L_z = self.angular_momentum_field[:, :, k, 2]
                
                # 只考虑正的角动量z分量
                positive_L_z = np.maximum(L_z, 0)
                total_weight = np.sum(positive_L_z)
                
                if total_weight > 0:
                    # 创建坐标网格（注意索引顺序）
                    X_plane, Y_plane = np.meshgrid(self.grid_x, self.grid_y, indexing='ij')
                    
                    # 计算加权中心
                    center_x = np.sum(X_plane * positive_L_z) / total_weight
                    center_y = np.sum(Y_plane * positive_L_z) / total_weight
                    center_z = self.grid_z[k]
                    
                    axis_points.append([center_x, center_y, center_z])
                else:
                    # 使用光束中心 / Use beam center
                    axis_points.append([self.center[0], self.center[1], self.grid_z[k]])
            
            axis_points = np.array(axis_points)
            
            # 改进的轴方向计算 / Improved axis direction calculation
            if len(axis_points) > 2:
                # 使用线性回归拟合，更稳定
                z_coords = axis_points[:, 2]
                x_coords = axis_points[:, 0]
                y_coords = axis_points[:, 1]
                
                # 计算z方向的梯度 / Calculate gradient in z direction
                if len(set(z_coords)) > 1:  # 确保z坐标不全相同 / Ensure z coordinates are not all the same
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
            print(f"警告：角动量轴计算失败 / Warning: Angular momentum axis calculation failed: {e}")
            self.axis_points = np.array([self.center])
            self.axis_direction = np.array([0, 0, 1])
            return self.axis_points, self.axis_direction

    def calculate_torque_at_position(self, position, particle_radius=1e-6, refractive_index=1.5):
        """改进的扭矩计算 / Improved torque calculation"""
        # 获取角动量密度 / Get angular momentum density
        L = self.get_angular_momentum_at_position(position)
        
        # 计算散射截面（Rayleigh散射） / Calculate scattering cross-section (Rayleigh scattering)
        k = 2 * np.pi / self.wavelength
        x = k * particle_radius  # 尺寸参数 / Size parameter
        
        if x < 0.1:  # Rayleigh散射区域 / Rayleigh scattering region
            σ_sca = (8 * pi / 3) * (k**4) * (particle_radius**6) * \
                    ((refractive_index**2 - 1) / (refractive_index**2 + 2))**2
        else:
            # 使用Mie散射或几何光学近似 / Use Mie scattering or geometric optics approximation
            σ_sca = np.pi * particle_radius**2  # 简化为几何截面 / Simplified to geometric cross-section
        
        # 计算角动量转移率 / Calculate angular momentum transfer rate
        ω = 2 * np.pi * c / self.wavelength
        L_magnitude = np.linalg.norm(L)
        
        # 扭矩 = 角动量转移率 / Torque = angular momentum transfer rate
        torque_magnitude = σ_sca * L_magnitude * c / ω
        
        # 扭矩方向沿角动量方向 / Torque direction along angular momentum direction
        if L_magnitude > 0:
            torque = torque_magnitude * L / L_magnitude
        else:
            torque = np.zeros(3)
        
        return torque

    def calculate_torque_field(self):
        """计算扭矩密度场 / Calculate torque density field
        τ = ∇ × L，其中L是角动量密度 / τ = ∇ × L, where L is angular momentum density
        """
        if self.angular_momentum_field is None:
            self.calculate_angular_momentum_field()
        
        # 计算角动量场的旋度 / Calculate curl of angular momentum field
        L_x = self.angular_momentum_field[..., 0]
        L_y = self.angular_momentum_field[..., 1] 
        L_z = self.angular_momentum_field[..., 2]
        
        # 使用有限差分计算旋度 / Use finite difference to calculate curl
        dx = self.grid_x[1] - self.grid_x[0]
        dy = self.grid_y[1] - self.grid_y[0]
        dz = self.grid_z[1] - self.grid_z[0]
        
        # τ_x = ∂L_z/∂y - ∂L_y/∂z
        tau_x = np.gradient(L_z, dy, axis=1) - np.gradient(L_y, dz, axis=2)
        # τ_y = ∂L_x/∂z - ∂L_z/∂x  
        tau_y = np.gradient(L_x, dz, axis=2) - np.gradient(L_z, dx, axis=0)
        # τ_z = ∂L_y/∂x - ∂L_x/∂y
        tau_z = np.gradient(L_y, dx, axis=0) - np.gradient(L_x, dy, axis=1)
        
        self.torque_field = np.stack([tau_x, tau_y, tau_z], axis=-1)
        return self.torque_field