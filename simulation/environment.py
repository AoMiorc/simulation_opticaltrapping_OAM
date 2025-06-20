import numpy as np
from scipy.constants import Boltzmann as k_B

class Environment:
    """表示粒子所在的环境（液体或气体） / Represents the environment (liquid or gas) where particles exist"""
    def __init__(self, medium='liquid', T=298.0, eta=0.001, P_gas=101325.0, M_gas=4.8e-26):
        """
        初始化环境参数 / Initialize environment parameters
        
        参数 / Parameters:
        medium (str): 介质类型，'liquid'或'gas' / Medium type, 'liquid' or 'gas'
        T (float): 环境温度 (K) / Environment temperature (K)
        eta (float): 粘度 (Pa·s) / Viscosity (Pa·s)
        P_gas (float): 气体压力 (Pa)，仅对气体介质有效 / Gas pressure (Pa), only valid for gas medium
        M_gas (float): 气体分子质量 (kg)，仅对气体介质有效 / Gas molecular mass (kg), only valid for gas medium
        """
        self.medium = medium
        self.T = T  # 环境温度 (K) / Environment temperature (K)
        self.eta = eta  # 粘度 (Pa·s) / Viscosity (Pa·s)
        self.P_gas = P_gas  # 气体压力 (Pa) / Gas pressure (Pa)
        self.M_gas = M_gas  # 气体分子质量 (kg) / Gas molecular mass (kg)
    
    def get_drag_coefficient(self, particle):
        """计算阻尼系数γ_q，根据介质类型使用不同的公式 / Calculate drag coefficient γ_q using different formulas based on medium type"""
        a = particle.radius
        
        if self.medium == 'liquid':
            # 液体环境：斯托克斯定律 / Liquid environment: Stokes' law
            return 6 * np.pi * a * self.eta
        
        elif self.medium == 'gas':
            # 气体环境：计算克努森数和阻尼率 / Gas environment: calculate Knudsen number and damping rate
            # 计算平均自由程 / Calculate mean free path
            mean_free_path = (self.eta / self.P_gas) * np.sqrt(np.pi * k_B * self.T / (2 * self.M_gas))
            
            # 克努森数 / Knudsen number
            Kn = mean_free_path / a
            
            # 完整阻尼率公式 / Complete damping rate formula
            term1 = 0.619 / (0.619 + Kn)
            term2 = 1 + (0.31 * Kn) / (0.785 + 1.152 * Kn + Kn**2)
            Gamma_q = (6 * np.pi * a * self.eta / particle.mass) * term1 * term2
            
            # 对于低压力情况使用近似公式 / Use approximation formula for low pressure cases
            if Kn > 10:  # Kn >> 1
                Gamma_q = 3.714 * (a**2 / particle.mass) * np.sqrt(
                    2 * np.pi * self.M_gas / (k_B * self.T)) * self.P_gas
            
            return Gamma_q * particle.mass  # 转换为阻尼系数γ_q = Γ_q * M / Convert to drag coefficient γ_q = Γ_q * M
        
        else:
            raise ValueError(f"未知介质类型 / Unknown medium type: {self.medium}")
    
    def get_angular_drag_coefficient(self, particle):
        """计算角阻尼系数γ_rot，根据介质类型使用不同的公式 / Calculate angular drag coefficient γ_rot using different formulas based on medium type"""
        a = particle.radius
        
        if self.medium == 'liquid':
            # 液体环境：γrot = 8πa³η / Liquid environment: γrot = 8πa³η
            return 8 * np.pi * a**3 * self.eta
        
        elif self.medium == 'gas':
            # 气体环境：γrot = 8πa³η * (P_gas/101325) / Gas environment: γrot = 8πa³η * (P_gas/101325)
            return 8 * np.pi * a**3 * self.eta * (self.P_gas/101325)
        
        else:
            raise ValueError(f"未知介质类型 / Unknown medium type: {self.medium}")