import numpy as np

class Particle:
    """表示被捕获的粒子 / Represents a trapped particle"""
    def __init__(self, mass, radius, position=None):
        """初始化粒子 / Initialize particle
        
        参数 / Parameters:
        mass (float): 粒子质量 (kg) / Particle mass (kg)
        radius (float): 粒子半径 (m) / Particle radius (m)
        position (np.array): 初始位置 (m)，默认为原点 / Initial position (m), defaults to origin
        """
        self.mass = mass
        self.radius = radius
        self.position = np.array([0.0, 0.0, 0.0]) if position is None else position
        self.velocity = np.array([0.0, 0.0, 0.0])  # 初始速度 / Initial velocity
        self.acceleration = np.array([0.0, 0.0, 0.0])  # 初始加速度 / Initial acceleration
        self.force = np.array([0.0, 0.0, 0.0])  # 当前受力 / Current force
        
        # 添加角运动相关属性 / Add angular motion related properties
        self.moment_of_inertia = (2/5) * mass * radius**2  # 球形粒子的转动惯量 / Moment of inertia for spherical particle
        self.angular_velocity = np.array([0.0, 0.0, 0.0])  # 角速度 / Angular velocity
        self.angular_acceleration = np.array([0.0, 0.0, 0.0])  # 角加速度 / Angular acceleration
        self.torque = np.array([0.0, 0.0, 0.0])  # 当前扭矩 / Current torque
        self.orientation = np.array([0.0, 0.0, 0.0])  # 初始化欧拉角姿态 / Initialize Euler angle orientation


class ParticleFactory:
    """粒子工厂类，用于创建不同类型的粒子 / Particle factory class for creating different types of particles"""
    
    # 预定义的粒子材料属性 / Predefined particle material properties
    MATERIALS = {
        'polystyrene': {'density': 1050},  # kg/m³
        'silica': {'density': 2200},       # kg/m³
        'gold': {'density': 19300},        # kg/m³
        'silver': {'density': 10490},      # kg/m³
        'latex': {'density': 1020},        # kg/m³
        'pmma': {'density': 1180},         # kg/m³
        'glass': {'density': 2500},        # kg/m³
    }
    
    @classmethod
    def create_particle(cls, material, radius, position=None):
        """根据材料和半径创建粒子 / Create particle based on material and radius
        
        参数 / Parameters:
        material (str): 材料名称 / Material name
        radius (float): 粒子半径 (m) / Particle radius (m)
        position (np.array): 初始位置 (m) / Initial position (m)
        
        返回 / Returns:
        Particle: 创建的粒子对象 / Created particle object
        """
        if material not in cls.MATERIALS:
            raise ValueError(f"未知材料 / Unknown material: {material}. 可用材料 / Available materials: {list(cls.MATERIALS.keys())}")
        
        density = cls.MATERIALS[material]['density']
        volume = (4/3) * np.pi * radius**3
        mass = density * volume
        
        return Particle(mass, radius, position)
    
    @classmethod
    def create_custom_particle(cls, mass, radius, position=None):
        """创建自定义质量的粒子 / Create particle with custom mass
        
        参数 / Parameters:
        mass (float): 粒子质量 (kg) / Particle mass (kg)
        radius (float): 粒子半径 (m) / Particle radius (m)
        position (np.array): 初始位置 (m) / Initial position (m)
        
        返回 / Returns:
        Particle: 创建的粒子对象 / Created particle object
        """
        return Particle(mass, radius, position)
    
    @classmethod
    def create_polystyrene_sphere(cls, radius, position=None):
        """创建聚苯乙烯球形粒子（常用于光镊实验） / Create polystyrene spherical particle (commonly used in optical tweezers experiments)
        
        参数 / Parameters:
        radius (float): 粒子半径 (m) / Particle radius (m)
        position (np.array): 初始位置 (m) / Initial position (m)
        
        返回 / Returns:
        Particle: 创建的粒子对象 / Created particle object
        """
        return cls.create_particle('polystyrene', radius, position)
    
    @classmethod
    def create_silica_sphere(cls, radius, position=None):
        """创建二氧化硅球形粒子 / Create silica spherical particle
        
        参数 / Parameters:
        radius (float): 粒子半径 (m) / Particle radius (m)
        position (np.array): 初始位置 (m) / Initial position (m)
        
        返回 / Returns:
        Particle: 创建的粒子对象 / Created particle object
        """
        return cls.create_particle('silica', radius, position)
    
    @classmethod
    def create_gold_nanoparticle(cls, radius, position=None):
        """创建金纳米粒子 / Create gold nanoparticle
        
        参数 / Parameters:
        radius (float): 粒子半径 (m) / Particle radius (m)
        position (np.array): 初始位置 (m) / Initial position (m)
        
        返回 / Returns:
        Particle: 创建的粒子对象 / Created particle object
        """
        return cls.create_particle('gold', radius, position)
    
    @classmethod
    def create_multiple_particles(cls, material, radius, positions):
        """批量创建相同材料和大小的粒子 / Batch create particles with same material and size
        
        参数 / Parameters:
        material (str): 材料名称 / Material name
        radius (float): 粒子半径 (m) / Particle radius (m)
        positions (list): 位置列表，每个元素为np.array / Position list, each element is np.array
        
        返回 / Returns:
        list: 粒子对象列表 / List of particle objects
        """
        particles = []
        for position in positions:
            particles.append(cls.create_particle(material, radius, position))
        return particles
    
    @classmethod
    def create_random_particles(cls, material, radius, num_particles, 
                              x_range=(-1e-6, 1e-6), y_range=(-1e-6, 1e-6), z_range=(-1e-6, 1e-6)):
        """创建随机分布的粒子 / Create randomly distributed particles
        
        参数 / Parameters:
        material (str): 材料名称 / Material name
        radius (float): 粒子半径 (m) / Particle radius (m)
        num_particles (int): 粒子数量 / Number of particles
        x_range (tuple): x坐标范围 (m) / x coordinate range (m)
        y_range (tuple): y坐标范围 (m) / y coordinate range (m)
        z_range (tuple): z坐标范围 (m) / z coordinate range (m)
        
        返回 / Returns:
        list: 粒子对象列表 / List of particle objects
        """
        particles = []
        for _ in range(num_particles):
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            z = np.random.uniform(z_range[0], z_range[1])
            position = np.array([x, y, z])
            particles.append(cls.create_particle(material, radius, position))
        return particles
    
    @classmethod
    def add_material(cls, name, density):
        """添加新的材料类型 / Add new material type
        
        参数 / Parameters:
        name (str): 材料名称 / Material name
        density (float): 材料密度 (kg/m³) / Material density (kg/m³)
        """
        cls.MATERIALS[name] = {'density': density}
    
    @classmethod
    def get_available_materials(cls):
        """获取可用的材料列表 / Get list of available materials
        
        返回 / Returns:
        list: 可用材料名称列表 / List of available material names
        """
        return list(cls.MATERIALS.keys())