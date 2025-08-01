import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import os
from scipy.interpolate import RegularGridInterpolator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ParticleFieldVideoGenerator:
    def __init__(self, trajectory_csv, field_csv, field_size_cm=2.0):
        """
        初始化视频生成器
        
        Args:
            trajectory_csv: 粒子轨迹CSV文件路径
            field_csv: 光场强度CSV文件路径
            field_size_cm: 光场尺寸（厘米）
        """
        self.trajectory_csv = trajectory_csv
        self.field_csv = field_csv
        self.field_size_cm = field_size_cm
        self.field_size_m = field_size_cm / 100.0
        
        # 加载数据
        self.load_trajectory_data()
        self.load_field_data()
        
    def load_trajectory_data(self):
        """加载粒子轨迹数据"""
        print(f"正在加载轨迹数据: {self.trajectory_csv}")
        self.trajectory_data = pd.read_csv(self.trajectory_csv)
        
        # 转换单位为微米
        self.trajectory_data['X_um'] = self.trajectory_data['X (m)'] * 1e6
        self.trajectory_data['Y_um'] = self.trajectory_data['Y (m)'] * 1e6
        self.trajectory_data['Z_um'] = self.trajectory_data['Z (m)'] * 1e6
        
        print(f"轨迹数据加载完成，共 {len(self.trajectory_data)} 个数据点")
        
    def load_field_data(self):
        """加载光场强度数据"""
        print(f"正在加载光场数据: {self.field_csv}")
        
        try:
            # 加载CSV数据
            self.intensity_data = np.loadtxt(self.field_csv, delimiter=',')
            print(f"光场数据加载完成，数据形状: {self.intensity_data.shape}")
            
            # 创建坐标网格
            if len(self.intensity_data.shape) == 2:
                # 2D数据，假设是XY平面
                ny, nx = self.intensity_data.shape
                
                # 创建坐标轴（微米单位）
                self.x_field = np.linspace(-self.field_size_m/2*1e6, self.field_size_m/2*1e6, nx)
                self.y_field = np.linspace(-self.field_size_m/2*1e6, self.field_size_m/2*1e6, ny)
                
                # 创建网格
                self.X_field, self.Y_field = np.meshgrid(self.x_field, self.y_field)
                
                print(f"光场范围: X=[{self.x_field[0]:.1f}, {self.x_field[-1]:.1f}] μm")
                print(f"光场范围: Y=[{self.y_field[0]:.1f}, {self.y_field[-1]:.1f}] μm")
                
            else:
                raise ValueError(f"不支持的数据格式: {self.intensity_data.shape}")
                
        except Exception as e:
            print(f"加载光场数据失败: {e}")
            # 创建默认的光场数据
            self.create_default_field()
    
    def create_default_field(self):
        """创建默认的光场数据（如果加载失败）"""
        print("创建默认光场数据...")
        
        # 创建网格
        nx, ny = 200, 200
        self.x_field = np.linspace(-self.field_size_m/2*1e6, self.field_size_m/2*1e6, nx)
        self.y_field = np.linspace(-self.field_size_m/2*1e6, self.field_size_m/2*1e6, ny)
        self.X_field, self.Y_field = np.meshgrid(self.x_field, self.y_field)
        
        # 创建LP71模式的近似光场
        r = np.sqrt(self.X_field**2 + self.Y_field**2)
        theta = np.arctan2(self.Y_field, self.X_field)
        
        # LP71模式的近似表达式
        w0 = self.field_size_m/4 * 1e6  # 束腰半径（微米）
        self.intensity_data = (r/w0)**14 * np.exp(-2*(r/w0)**2) * np.cos(7*theta)**2
        
        print("默认光场数据创建完成")
    
    def create_video(self, output_filename='particle_field_video.mp4', 
                    fps=30, duration_sec=10, trail_length=50):
        """
        创建粒子轨迹视频
        
        Args:
            output_filename: 输出视频文件名
            fps: 帧率
            duration_sec: 视频时长（秒）
            trail_length: 轨迹尾迹长度
        """
        print(f"开始生成视频: {output_filename}")
        
        # 计算总帧数
        total_frames = fps * duration_sec
        
        # 计算每帧对应的数据点数
        data_points_per_frame = len(self.trajectory_data) // total_frames
        if data_points_per_frame < 1:
            data_points_per_frame = 1
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制光场背景
        im = ax.imshow(self.intensity_data, 
                      extent=[self.x_field[0], self.x_field[-1], 
                             self.y_field[0], self.y_field[-1]],
                      origin='lower', 
                      cmap='hot', 
                      alpha=0.8,
                      norm=LogNorm(vmin=np.max(self.intensity_data)*1e-6, 
                                  vmax=np.max(self.intensity_data)))
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('光场强度 (相对单位)', fontsize=12)
        
        # 设置坐标轴
        ax.set_xlabel('X位置 (μm)', fontsize=12)
        ax.set_ylabel('Y位置 (μm)', fontsize=12)
        ax.set_title('粒子在光场中的运动轨迹', fontsize=14)
        
        # 初始化轨迹线和粒子点
        trail_line, = ax.plot([], [], 'cyan', linewidth=2, alpha=0.7, label='轨迹')
        particle_point, = ax.plot([], [], 'wo', markersize=8, markeredgecolor='red', 
                                 markeredgewidth=2, label='粒子')
        
        # 添加时间文本
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 添加图例
        ax.legend(loc='upper right')
        
        # 设置坐标轴范围
        x_range = [self.trajectory_data['X_um'].min() - 1, 
                  self.trajectory_data['X_um'].max() + 1]
        y_range = [self.trajectory_data['Y_um'].min() - 1, 
                  self.trajectory_data['Y_um'].max() + 1]
        
        # 确保显示范围包含光场范围
        x_range[0] = min(x_range[0], self.x_field[0])
        x_range[1] = max(x_range[1], self.x_field[-1])
        y_range[0] = min(y_range[0], self.y_field[0])
        y_range[1] = max(y_range[1], self.y_field[-1])
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        def animate(frame):
            """动画函数"""
            # 计算当前数据点索引
            current_idx = min(frame * data_points_per_frame, len(self.trajectory_data) - 1)
            
            # 计算轨迹尾迹的起始索引
            trail_start = max(0, current_idx - trail_length)
            
            # 获取轨迹数据
            if current_idx > 0:
                x_trail = self.trajectory_data['X_um'].iloc[trail_start:current_idx+1]
                y_trail = self.trajectory_data['Y_um'].iloc[trail_start:current_idx+1]
                
                # 更新轨迹线
                trail_line.set_data(x_trail, y_trail)
                
                # 更新粒子位置
                x_current = self.trajectory_data['X_um'].iloc[current_idx]
                y_current = self.trajectory_data['Y_um'].iloc[current_idx]
                particle_point.set_data([x_current], [y_current])
                
                # 更新时间显示
                current_time = self.trajectory_data['Time (s)'].iloc[current_idx]
                time_text.set_text(f'时间: {current_time:.4f} s\n帧: {frame+1}/{total_frames}')
            
            return trail_line, particle_point, time_text
        
        # 创建动画
        print(f"正在创建动画，总帧数: {total_frames}")
        anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                     interval=1000//fps, blit=True, repeat=True)
        
        # 保存视频
        print(f"正在保存视频到: {output_filename}")
        try:
            # 尝试使用ffmpeg编码器
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='ParticleFieldVideo'), bitrate=1800)
            anim.save(output_filename, writer=writer)
            print(f"视频保存成功: {output_filename}")
        except Exception as e:
            print(f"使用ffmpeg保存失败: {e}")
            try:
                # 尝试使用pillow保存为GIF
                gif_filename = output_filename.replace('.mp4', '.gif')
                anim.save(gif_filename, writer='pillow', fps=fps//2)
                print(f"已保存为GIF格式: {gif_filename}")
            except Exception as e2:
                print(f"保存失败: {e2}")
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def create_high_quality_video(self, output_filename='particle_field_hq_video.mp4',
                                 fps=60, duration_sec=15, trail_length=100):
        """
        创建高质量视频
        """
        print("创建高质量视频...")
        
        # 使用更高的DPI和更好的设置
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
        
        return self.create_video(output_filename, fps, duration_sec, trail_length)
    
    def preview_field_and_trajectory(self):
        """
        预览光场和轨迹的静态图
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制光场背景
        im = ax.imshow(self.intensity_data, 
                      extent=[self.x_field[0], self.x_field[-1], 
                             self.y_field[0], self.y_field[-1]],
                      origin='lower', 
                      cmap='hot', 
                      alpha=0.6,
                      norm=LogNorm(vmin=np.max(self.intensity_data)*1e-6, 
                                  vmax=np.max(self.intensity_data)))
        
        # 绘制完整轨迹
        ax.plot(self.trajectory_data['X_um'], self.trajectory_data['Y_um'], 
               'cyan', linewidth=2, alpha=0.8, label='粒子轨迹')
        
        # 标记起点和终点
        ax.plot(self.trajectory_data['X_um'].iloc[0], self.trajectory_data['Y_um'].iloc[0], 
               'go', markersize=10, label='起点')
        ax.plot(self.trajectory_data['X_um'].iloc[-1], self.trajectory_data['Y_um'].iloc[-1], 
               'ro', markersize=10, label='终点')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('光场强度 (相对单位)', fontsize=12)
        
        # 设置标签和标题
        ax.set_xlabel('X位置 (μm)', fontsize=12)
        ax.set_ylabel('Y位置 (μm)', fontsize=12)
        ax.set_title('光场中的粒子轨迹预览', fontsize=14)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('field_trajectory_preview.png', dpi=300, bbox_inches='tight')
        plt.show()

# 主函数
def main():
    # 文件路径
    trajectory_file = 'particle_trajectory_lp71_csv_new.csv'
    field_file = 'final_intensity_LP71_minus6_20cm.csv'
    
    # 检查文件是否存在
    if not os.path.exists(trajectory_file):
        print(f"错误: 找不到轨迹文件 {trajectory_file}")
        return
    
    if not os.path.exists(field_file):
        print(f"警告: 找不到光场文件 {field_file}，将使用默认光场")
    
    # 创建视频生成器
    video_gen = ParticleFieldVideoGenerator(trajectory_file, field_file, field_size_cm=2.0)
    
    # 预览静态图
    print("生成预览图...")
    video_gen.preview_field_and_trajectory()
    
    # 生成标准质量视频
    print("\n生成标准质量视频...")
    anim1 = video_gen.create_video('particle_field_video.mp4', fps=30, duration_sec=10)
    
    # 生成高质量视频
    print("\n生成高质量视频...")
    anim2 = video_gen.create_high_quality_video('particle_field_hq_video.mp4', fps=60, duration_sec=15)
    
    print("\n视频生成完成！")
    print("生成的文件:")
    print("- field_trajectory_preview.png (预览图)")
    print("- particle_field_video.mp4 (标准质量视频)")
    print("- particle_field_hq_video.mp4 (高质量视频)")

if __name__ == "__main__":
    main()