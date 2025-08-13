import numpy as np
from scipy.constants import Boltzmann as k_B

class Environment:
    """Represents the environment (liquid or gas) where particles exist"""
    def __init__(self, medium='liquid', T=298.0, eta=0.001, P_gas=101325.0, M_gas=4.8e-26, refractive_index=1.33):
        """
        Initialize environment parameters
        
        Parameters:
        medium (str): Medium type, 'liquid' or 'gas'
        T (float): Environment temperature (K)
        eta (float): Viscosity (Pa·s)
        P_gas (float): Gas pressure (Pa), only valid for gas medium
        M_gas (float): Gas molecular mass (kg), only valid for gas medium
        refractive_index (float): Medium refractive index, defaults to 1.33 (water)
        """
        self.medium = medium
        self.T = T  # Environment temperature (K)
        self.eta = eta  # Viscosity (Pa·s)
        self.P_gas = P_gas  # Gas pressure (Pa)
        self.M_gas = M_gas  # Gas molecular mass (kg)
        self.refractive_index = refractive_index  # Medium refractive index

    
    def get_drag_coefficient(self, particle):
        """
        Calculate drag coefficient γ_q using different formulas based on medium type
        
        Parameters:
        particle: Particle object
        
        Returns:
        float: Drag coefficient γ_q
        """
        a = particle.radius
        
        if self.medium == 'liquid':
            # Liquid environment: Stokes' law
            return 6 * np.pi * a * self.eta
        
        elif self.medium == 'gas':
            # Gas environment: calculate Knudsen number and damping rate
            # Calculate mean free path
            mean_free_path = (self.eta / self.P_gas) * np.sqrt(np.pi * k_B * self.T / (2 * self.M_gas))
            
            # Knudsen number
            Kn = mean_free_path / a
            
            # Complete damping rate formula
            term1 = 0.619 / (0.619 + Kn)
            term2 = 1 + (0.31 * Kn) / (0.785 + 1.152 * Kn + Kn**2)
            Gamma_q = (6 * np.pi * a * self.eta / particle.mass) * term1 * term2
            
            # Use approximation formula for low pressure cases
            if Kn > 10:  # Kn >> 1
                Gamma_q = 3.714 * (a**2 / particle.mass) * np.sqrt(
                    2 * np.pi * self.M_gas / (k_B * self.T)) * self.P_gas
            
            return Gamma_q * particle.mass  # γ_q = Γ_q * m / Convert to drag coefficient γ_q = Γ_q * m
        
        else:
            raise ValueError(f"Unknown medium type: {self.medium}")
    
    def get_angular_drag_coefficient(self, particle):
        """
        Calculate angular drag coefficient γ_rot using different formulas based on medium type
        
        Parameters:
        particle: Particle object
        
        Returns:
        float: Angular drag coefficient γ_rot
        """
        a = particle.radius
        
        if self.medium == 'liquid':
            # γrot = 8πa³η / Liquid environment: γrot = 8πa³η
            return 8 * np.pi * a**3 * self.eta
        
        elif self.medium == 'gas':
            # γrot = 8πa³η * (P_gas/101325) / Gas environment: γrot = 8πa³η * (P_gas/101325)
            return 8 * np.pi * a**3 * self.eta * (self.P_gas/101325)
        
        else:
            raise ValueError(f"Unknown medium type: {self.medium}")