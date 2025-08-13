import numpy as np

class Particle:
    """Represents a trapped particle"""
    def __init__(self, mass, radius, position=None, refractive_index=1.59):
        """
        Initialize particle
        
        Parameters:
        mass (float): Particle mass (kg)
        radius (float): Particle radius (m)
        position (np.array): Initial position (m), defaults to origin (0, 0, 0)
        refractive_index (float): Particle refractive index, defaults to 1.59 (polystyrene)
        """
        self.mass = mass
        self.radius = radius
        self.refractive_index = refractive_index  # 添加折射率属性
        self.position = np.array([0.0, 0.0, 0.0]) if position is None else position
        self.velocity = np.array([0.0, 0.0, 0.0])  # Initial velocity
        self.acceleration = np.array([0.0, 0.0, 0.0])  # Initial acceleration
        self.force = np.array([0.0, 0.0, 0.0])  # Current force
        
        # angular motion related properties
        self.moment_of_inertia = (2/5) * mass * radius**2  # Moment of inertia for spherical particle
        self.angular_velocity = np.array([0.0, 0.0, 0.0])  # Angular velocity
        self.angular_acceleration = np.array([0.0, 0.0, 0.0])  # Angular acceleration   
        self.torque = np.array([0.0, 0.0, 0.0])  # Current torque
        self.orientation = np.array([0.0, 0.0, 0.0])  # Initialize Euler angle orientation


class ParticleFactory:
    """Particle factory class for creating different types of particles"""
    
    # Predefined particle material properties
    MATERIALS = {
        'polystyrene': {'density': 1050, 'refractive_index': 1.59},  # kg/m³
        'silica': {'density': 2200, 'refractive_index': 1.46},       # kg/m³
        'gold': {'density': 19300, 'refractive_index': 0.47},        # kg/m³ 
        'silver': {'density': 10490, 'refractive_index': 0.05},      # kg/m³ 
        'latex': {'density': 1020, 'refractive_index': 1.59},        # kg/m³
        'pmma': {'density': 1180, 'refractive_index': 1.49},         # kg/m³
        'glass': {'density': 2500, 'refractive_index': 1.52},        # kg/m³
    }
    
    @classmethod
    def create_particle(cls, material, radius, position=None):
        """Create particle based on material and radius
        
        Parameters:
        material (str): Material name
        radius (float): Particle radius (m)
        position (np.array): Initial position (m)
        
        Returns:
        Particle: Created particle object
        """
        if material not in cls.MATERIALS:
            raise ValueError(f"Unknown material: {material}. Available materials: {list(cls.MATERIALS.keys())}")
        
        density = cls.MATERIALS[material]['density']
        refractive_index = cls.MATERIALS[material]['refractive_index']
        volume = (4/3) * np.pi * radius**3
        mass = density * volume
        
        return Particle(mass, radius, position, refractive_index)
    
    @classmethod
    def create_custom_particle(cls, mass, radius, position=None):
        """Create particle with custom mass
        
        Parameters:
        mass (float): Particle mass (kg)
        radius (float): Particle radius (m)
        position (np.array): Initial position (m)
        
        Returns:
        Particle: Created particle object
        """
        return Particle(mass, radius, position)
    
    @classmethod
    def create_polystyrene_sphere(cls, radius, position=None):
        """Create polystyrene spherical particle (commonly used in optical tweezers experiments)
        
        Parameters:
        radius (float): Particle radius (m)
        position (np.array): Initial position (m)
        
        Returns:
        Particle: Created particle object
        """
        return cls.create_particle('polystyrene', radius, position)
    
    @classmethod
    def create_silica_sphere(cls, radius, position=None):
        """Create silica spherical particle
        
        Parameters:
        radius (float): Particle radius (m)
        position (np.array): Initial position (m)
        
        Returns:
        Particle: Created particle object
        """
        return cls.create_particle('silica', radius, position)
    
    @classmethod
    def create_gold_nanoparticle(cls, radius, position=None):
        """Create gold nanoparticle
        
        Parameters:
        radius (float): Particle radius (m)
        position (np.array): Initial position (m)
        
        Returns:
        Particle: Created particle object
        """
        return cls.create_particle('gold', radius, position)
    
    @classmethod
    def create_multiple_particles(cls, material, radius, positions):
        """Batch create particles with same material and size
        
        Parameters:
        material (str): Material name
        radius (float): Particle radius (m)
        positions (list): Position list, each element is np.array
        
        Returns:
        list: List of particle objects
        """
        particles = []
        for position in positions:
            particles.append(cls.create_particle(material, radius, position))
        return particles
    
    @classmethod
    def create_random_particles(cls, material, radius, num_particles, 
                              x_range=(-1e-6, 1e-6), y_range=(-1e-6, 1e-6), z_range=(-1e-6, 1e-6)):
        """Batch create randomly distributed particles
        
        Parameters:
        material (str): Material name
        radius (float): Particle radius (m)
        num_particles (int): Number of particles
        x_range (tuple): x coordinate range (m)
        y_range (tuple): y coordinate range (m)
        z_range (tuple): z coordinate range (m)
        
        Returns:
        list: List of particle objects
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
    def add_material(cls, name, density, refractive_index):
        """Add new material type
        
        Parameters:
        name (str): Material name
        density (float): Material density (kg/m³)
        refractive_index (float): Material refractive index
        """
        cls.MATERIALS[name] = {'density': density, 'refractive_index': refractive_index}
    
    @classmethod
    def get_available_materials(cls):
        """Get list of available materials
        
        Returns:
        list: List of available material names
        """
        return list(cls.MATERIALS.keys())