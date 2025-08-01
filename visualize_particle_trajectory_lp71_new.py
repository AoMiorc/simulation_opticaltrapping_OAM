import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.animation import FuncAnimation

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class ParticleTrajectoryVisualizer:
    def __init__(self, csv_file):
        """初始化可视化器"""
        self.data = pd.read_csv(csv_file)
        self.setup_data()
    
    def setup_data(self):
        """数据预处理"""
        # 转换单位以便更好地显示
        self.data['X_um'] = self.data['X (m)'] * 1e6  # 转换为微米
        self.data['Y_um'] = self.data['Y (m)'] * 1e6
        self.data['Z_um'] = self.data['Z (m)'] * 1e6
        
        # 计算速度和力的大小
        self.data['Speed'] = np.sqrt(self.data['Vx (m/s)']**2 + 
                                   self.data['Vy (m/s)']**2 + 
                                   self.data['Vz (m/s)']**2)
        
        self.data['Force_magnitude'] = np.sqrt(self.data['Fx (N)']**2 + 
                                             self.data['Fy (N)']**2 + 
                                             self.data['Fz (N)']**2)
        
        self.data['Torque_magnitude'] = np.sqrt(self.data['τx (pN·μm)']**2 + 
                                              self.data['τy (pN·μm)']**2 + 
                                              self.data['τz (pN·μm)']**2)
    
    def plot_3d_trajectory(self):
        """绘制3D轨迹图"""
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 根据时间着色
        scatter = ax.scatter(self.data['X_um'], self.data['Y_um'], self.data['Z_um'],
                           c=self.data['Time (s)'], cmap='viridis', s=20, alpha=0.7)
        
        ax.set_xlabel('X位置 (μm)')
        ax.set_ylabel('Y位置 (μm)')
        ax.set_zlabel('Z位置 (μm)')
        ax.set_title('粒子3D轨迹 (按时间着色)')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('时间 (s)')
        
        plt.tight_layout()
        plt.savefig('particle_3d_trajectory_lp71_new.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_2d_projections(self):
        """绘制2D投影图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # XY平面投影
        scatter1 = axes[0,0].scatter(self.data['X_um'], self.data['Y_um'], 
                                   c=self.data['Time (s)'], cmap='plasma', s=15, alpha=0.7)
        axes[0,0].set_xlabel('X位置 (μm)')
        axes[0,0].set_ylabel('Y位置 (μm)')
        axes[0,0].set_title('XY平面投影')
        plt.colorbar(scatter1, ax=axes[0,0])
        
        # XZ平面投影
        scatter2 = axes[0,1].scatter(self.data['X_um'], self.data['Z_um'], 
                                   c=self.data['Time (s)'], cmap='plasma', s=15, alpha=0.7)
        axes[0,1].set_xlabel('X位置 (μm)')
        axes[0,1].set_ylabel('Z位置 (μm)')
        axes[0,1].set_title('XZ平面投影')
        plt.colorbar(scatter2, ax=axes[0,1])
        
        # YZ平面投影
        scatter3 = axes[1,0].scatter(self.data['Y_um'], self.data['Z_um'], 
                                   c=self.data['Time (s)'], cmap='plasma', s=15, alpha=0.7)
        axes[1,0].set_xlabel('Y位置 (μm)')
        axes[1,0].set_ylabel('Z位置 (μm)')
        axes[1,0].set_title('YZ平面投影')
        plt.colorbar(scatter3, ax=axes[1,0])
        
        # 轨迹线图
        axes[1,1].plot(self.data['X_um'], self.data['Y_um'], 'b-', alpha=0.6, linewidth=1)
        axes[1,1].scatter(self.data['X_um'].iloc[0], self.data['Y_um'].iloc[0], 
                         c='green', s=100, marker='o', label='起点')
        axes[1,1].scatter(self.data['X_um'].iloc[-1], self.data['Y_um'].iloc[-1], 
                         c='red', s=100, marker='s', label='终点')
        axes[1,1].set_xlabel('X位置 (μm)')
        axes[1,1].set_ylabel('Y位置 (μm)')
        axes[1,1].set_title('XY轨迹线')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('particle_2d_projections_lp71_new.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_physical_quantities(self):
        """绘制物理量随时间变化"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # 位置随时间变化
        axes[0,0].plot(self.data['Time (s)'], self.data['X_um'], 'r-', label='X', alpha=0.8)
        axes[0,0].plot(self.data['Time (s)'], self.data['Y_um'], 'g-', label='Y', alpha=0.8)
        axes[0,0].plot(self.data['Time (s)'], self.data['Z_um'], 'b-', label='Z', alpha=0.8)
        axes[0,0].set_xlabel('时间 (s)')
        axes[0,0].set_ylabel('位置 (μm)')
        axes[0,0].set_title('位置随时间变化')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 速度大小随时间变化
        axes[0,1].plot(self.data['Time (s)'], self.data['Speed'], 'purple', linewidth=1.5)
        axes[0,1].set_xlabel('时间 (s)')
        axes[0,1].set_ylabel('速度大小 (m/s)')
        axes[0,1].set_title('速度大小随时间变化')
        axes[0,1].grid(True, alpha=0.3)
        
        # 力的分量
        axes[1,0].plot(self.data['Time (s)'], self.data['Fx (N)'], 'r-', label='Fx', alpha=0.8)
        axes[1,0].plot(self.data['Time (s)'], self.data['Fy (N)'], 'g-', label='Fy', alpha=0.8)
        axes[1,0].plot(self.data['Time (s)'], self.data['Fz (N)'], 'b-', label='Fz', alpha=0.8)
        axes[1,0].set_xlabel('时间 (s)')
        axes[1,0].set_ylabel('力 (N)')
        axes[1,0].set_title('力分量随时间变化')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 力的大小
        axes[1,1].plot(self.data['Time (s)'], self.data['Force_magnitude'], 'orange', linewidth=1.5)
        axes[1,1].set_xlabel('时间 (s)')
        axes[1,1].set_ylabel('力大小 (N)')
        axes[1,1].set_title('力大小随时间变化')
        axes[1,1].grid(True, alpha=0.3)
        
        # 角速度
        axes[2,0].plot(self.data['Time (s)'], self.data['ωz (rad/s)'], 'cyan', linewidth=1.5)
        axes[2,0].set_xlabel('时间 (s)')
        axes[2,0].set_ylabel('角速度 ωz (rad/s)')
        axes[2,0].set_title('Z轴角速度随时间变化')
        axes[2,0].grid(True, alpha=0.3)
        
        # 扭矩大小
        axes[2,1].plot(self.data['Time (s)'], self.data['Torque_magnitude'], 'brown', linewidth=1.5)
        axes[2,1].set_xlabel('时间 (s)')
        axes[2,1].set_ylabel('扭矩大小 (pN·μm)')
        axes[2,1].set_title('扭矩大小随时间变化')
        axes[2,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('particle_physical_quantities_lp71_new.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_phase_space(self):
        """绘制相空间图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # X位置-速度相空间
        scatter1 = axes[0,0].scatter(self.data['X_um'], self.data['Vx (m/s)'], 
                                   c=self.data['Time (s)'], cmap='coolwarm', s=10, alpha=0.7)
        axes[0,0].set_xlabel('X位置 (μm)')
        axes[0,0].set_ylabel('X速度 (m/s)')
        axes[0,0].set_title('X相空间 (位置-速度)')
        plt.colorbar(scatter1, ax=axes[0,0])
        
        # Y位置-速度相空间
        scatter2 = axes[0,1].scatter(self.data['Y_um'], self.data['Vy (m/s)'], 
                                   c=self.data['Time (s)'], cmap='coolwarm', s=10, alpha=0.7)
        axes[0,1].set_xlabel('Y位置 (μm)')
        axes[0,1].set_ylabel('Y速度 (m/s)')
        axes[0,1].set_title('Y相空间 (位置-速度)')
        plt.colorbar(scatter2, ax=axes[0,1])
        
        # Z位置-速度相空间
        scatter3 = axes[1,0].scatter(self.data['Z_um'], self.data['Vz (m/s)'], 
                                   c=self.data['Time (s)'], cmap='coolwarm', s=10, alpha=0.7)
        axes[1,0].set_xlabel('Z位置 (μm)')
        axes[1,0].set_ylabel('Z速度 (m/s)')
        axes[1,0].set_title('Z相空间 (位置-速度)')
        plt.colorbar(scatter3, ax=axes[1,0])
        
        # 角速度-扭矩关系
        scatter4 = axes[1,1].scatter(self.data['ωz (rad/s)'], self.data['τz (pN·μm)'], 
                                   c=self.data['Time (s)'], cmap='coolwarm', s=10, alpha=0.7)
        axes[1,1].set_xlabel('角速度 ωz (rad/s)')
        axes[1,1].set_ylabel('扭矩 τz (pN·μm)')
        axes[1,1].set_title('角速度-扭矩关系')
        plt.colorbar(scatter4, ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig('particle_phase_space_lp71_new.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_statistics_summary(self):
        """绘制统计摘要"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 位置分布直方图
        axes[0,0].hist(self.data['X_um'], bins=50, alpha=0.7, color='red', label='X')
        axes[0,0].hist(self.data['Y_um'], bins=50, alpha=0.7, color='green', label='Y')
        axes[0,0].hist(self.data['Z_um'], bins=50, alpha=0.7, color='blue', label='Z')
        axes[0,0].set_xlabel('位置 (μm)')
        axes[0,0].set_ylabel('频次')
        axes[0,0].set_title('位置分布直方图')
        axes[0,0].legend()
        
        # 速度分布
        axes[0,1].hist(self.data['Speed'], bins=50, alpha=0.7, color='purple')
        axes[0,1].set_xlabel('速度大小 (m/s)')
        axes[0,1].set_ylabel('频次')
        axes[0,1].set_title('速度大小分布')
        
        # 力分布
        axes[0,2].hist(self.data['Force_magnitude'], bins=50, alpha=0.7, color='orange')
        axes[0,2].set_xlabel('力大小 (N)')
        axes[0,2].set_ylabel('频次')
        axes[0,2].set_title('力大小分布')
        
        # 角速度分布
        axes[1,0].hist(self.data['ωz (rad/s)'], bins=50, alpha=0.7, color='cyan')
        axes[1,0].set_xlabel('角速度 ωz (rad/s)')
        axes[1,0].set_ylabel('频次')
        axes[1,0].set_title('角速度分布')
        
        # 扭矩分布
        axes[1,1].hist(self.data['Torque_magnitude'], bins=50, alpha=0.7, color='brown')
        axes[1,1].set_xlabel('扭矩大小 (pN·μm)')
        axes[1,1].set_ylabel('频次')
        axes[1,1].set_title('扭矩大小分布')
        
        # 能量相关（动能近似）
        kinetic_energy = 0.5 * self.data['Speed']**2  # 简化的动能（忽略质量）
        axes[1,2].plot(self.data['Time (s)'], kinetic_energy, 'darkgreen', linewidth=1.5)
        axes[1,2].set_xlabel('时间 (s)')
        axes[1,2].set_ylabel('动能 (相对单位)')
        axes[1,2].set_title('动能随时间变化')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('particle_statistics_lp71_new.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_animation(self, save_animation=False):
        """创建轨迹动画"""
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置坐标轴范围
        ax.set_xlim(self.data['X_um'].min(), self.data['X_um'].max())
        ax.set_ylim(self.data['Y_um'].min(), self.data['Y_um'].max())
        ax.set_zlim(self.data['Z_um'].min(), self.data['Z_um'].max())
        
        ax.set_xlabel('X位置 (μm)')
        ax.set_ylabel('Y位置 (μm)')
        ax.set_zlabel('Z位置 (μm)')
        ax.set_title('粒子轨迹动画')
        
        # 初始化线条和点
        line, = ax.plot([], [], [], 'b-', alpha=0.6, linewidth=2)
        point, = ax.plot([], [], [], 'ro', markersize=8)
        
        def animate(frame):
            # 显示到当前帧的轨迹
            end_idx = min(frame * 10, len(self.data))  # 每帧显示10个数据点
            
            if end_idx > 0:
                x_data = self.data['X_um'].iloc[:end_idx]
                y_data = self.data['Y_um'].iloc[:end_idx]
                z_data = self.data['Z_um'].iloc[:end_idx]
                
                line.set_data_3d(x_data, y_data, z_data)
                
                # 当前位置点
                if end_idx > 0:
                    point.set_data_3d([x_data.iloc[-1]], [y_data.iloc[-1]], [z_data.iloc[-1]])
            
            return line, point
        
        # 创建动画
        frames = len(self.data) // 10 + 1
        anim = FuncAnimation(fig, animate, frames=frames, interval=50, blit=False, repeat=True)
        
        if save_animation:
            anim.save('particle_trajectory_animation_lp71_new.gif', writer='pillow', fps=20)
            print("动画已保存为 particle_trajectory_animation_lp71_new.gif")
        
        plt.show()
        return anim
    
    def generate_report(self):
        """生成数据报告"""
        print("=" * 60)
        print("粒子轨迹数据分析报告")
        print("=" * 60)
        
        print(f"数据点总数: {len(self.data)}")
        print(f"时间范围: {self.data['Time (s)'].min():.6f} - {self.data['Time (s)'].max():.6f} 秒")
        print(f"时间步长: {self.data['Time (s)'].diff().mean():.6f} 秒")
        
        print("\n位置统计 (μm):")
        print(f"X: 范围 [{self.data['X_um'].min():.3f}, {self.data['X_um'].max():.3f}], 标准差 {self.data['X_um'].std():.3f}")
        print(f"Y: 范围 [{self.data['Y_um'].min():.3f}, {self.data['Y_um'].max():.3f}], 标准差 {self.data['Y_um'].std():.3f}")
        print(f"Z: 范围 [{self.data['Z_um'].min():.3f}, {self.data['Z_um'].max():.3f}], 标准差 {self.data['Z_um'].std():.3f}")
        
        print("\n速度统计:")
        print(f"平均速度大小: {self.data['Speed'].mean():.6e} m/s")
        print(f"最大速度: {self.data['Speed'].max():.6e} m/s")
        
        print("\n力统计:")
        print(f"平均力大小: {self.data['Force_magnitude'].mean():.6e} N")
        print(f"最大力: {self.data['Force_magnitude'].max():.6e} N")
        
        print("\n角速度统计:")
        print(f"平均角速度 ωz: {self.data['ωz (rad/s)'].mean():.3f} rad/s")
        print(f"角速度范围: [{self.data['ωz (rad/s)'].min():.3f}, {self.data['ωz (rad/s)'].max():.3f}] rad/s")
        
        print("\n扭矩统计:")
        print(f"平均扭矩大小: {self.data['Torque_magnitude'].mean():.6e} pN·μm")
        print(f"最大扭矩: {self.data['Torque_magnitude'].max():.6e} pN·μm")
        
        print("=" * 60)

# 主函数
def main():
    # 创建可视化器
    csv_file = 'particle_trajectory_lp71_csv_new.csv'
    visualizer = ParticleTrajectoryVisualizer(csv_file)
    
    # 生成报告
    visualizer.generate_report()
    
    # 创建各种图表
    print("\n正在生成3D轨迹图...")
    visualizer.plot_3d_trajectory()
    
    print("正在生成2D投影图...")
    visualizer.plot_2d_projections()
    
    print("正在生成物理量时间序列图...")
    visualizer.plot_physical_quantities()
    
    print("正在生成相空间图...")
    visualizer.plot_phase_space()
    
    print("正在生成统计摘要图...")
    visualizer.plot_statistics_summary()
    
    # 创建动画（可选）
    print("\n正在创建轨迹动画...")
    anim = visualizer.create_animation(save_animation=True)
    
    print("\n所有可视化图表已生成完成！")
    print("生成的文件:")
    print("- particle_3d_trajectory_lp71_new.png")
    print("- particle_2d_projections_lp71_new.png")
    print("- particle_physical_quantities_lp71_new.png")
    print("- particle_phase_space_lp71_new.png")
    print("- particle_statistics_lp71_new.png")
    print("- particle_trajectory_animation_lp71_new.gif")

if __name__ == "__main__":
    main()