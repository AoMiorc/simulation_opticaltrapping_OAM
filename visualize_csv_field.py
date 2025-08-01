import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def visualize_csv_field(csv_filename):
    """
    单独可视化CSV文件中的光场强度分布
    Visualize optical field intensity distribution from CSV file
    """
    try:
        # 加载CSV数据
        intensity_data = np.loadtxt(csv_filename, delimiter=',')
        print(f"成功加载光场数据，数据形状: {intensity_data.shape}")
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'LP71光场强度分布 - {csv_filename}', fontsize=16)
        
        # 1. 线性刻度显示
        im1 = axes[0, 0].imshow(intensity_data, cmap='hot', origin='lower')
        axes[0, 0].set_title('线性刻度 / Linear Scale')
        axes[0, 0].set_xlabel('X 像素 / X Pixel')
        axes[0, 0].set_ylabel('Y 像素 / Y Pixel')
        plt.colorbar(im1, ax=axes[0, 0], label='强度 / Intensity')
        
        # 2. 对数刻度显示（更好地显示多环结构）
        # 避免log(0)的问题
        intensity_log = intensity_data + np.max(intensity_data) * 1e-10
        im2 = axes[0, 1].imshow(intensity_log, cmap='hot', origin='lower', 
                               norm=LogNorm(vmin=np.min(intensity_log[intensity_log > 0])))
        axes[0, 1].set_title('对数刻度 / Log Scale')
        axes[0, 1].set_xlabel('X 像素 / X Pixel')
        axes[0, 1].set_ylabel('Y 像素 / Y Pixel')
        plt.colorbar(im2, ax=axes[0, 1], label='强度 (对数) / Intensity (Log)')
        
        # 3. 等高线图
        x = np.arange(intensity_data.shape[1])
        y = np.arange(intensity_data.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # 使用更多等高线层数来显示多环结构
        levels = 50
        contour = axes[1, 0].contourf(X, Y, intensity_data, levels=levels, cmap='hot')
        axes[1, 0].contour(X, Y, intensity_data, levels=levels, colors='white', alpha=0.3, linewidths=0.5)
        axes[1, 0].set_title('等高线图 / Contour Plot')
        axes[1, 0].set_xlabel('X 像素 / X Pixel')
        axes[1, 0].set_ylabel('Y 像素 / Y Pixel')
        plt.colorbar(contour, ax=axes[1, 0], label='强度 / Intensity')
        
        # 4. 中心截面图
        center_y = intensity_data.shape[0] // 2
        center_x = intensity_data.shape[1] // 2
        
        # X方向截面
        x_profile = intensity_data[center_y, :]
        # Y方向截面
        y_profile = intensity_data[:, center_x]
        
        x_coords = np.arange(len(x_profile))
        y_coords = np.arange(len(y_profile))
        
        axes[1, 1].plot(x_coords, x_profile, 'r-', linewidth=2, label='X方向截面 / X Profile')
        axes[1, 1].plot(y_coords, y_profile, 'b-', linewidth=2, label='Y方向截面 / Y Profile')
        axes[1, 1].set_title('中心截面强度分布 / Central Cross-Section')
        axes[1, 1].set_xlabel('像素位置 / Pixel Position')
        axes[1, 1].set_ylabel('强度 / Intensity')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 输出统计信息
        print("\n=== 光场数据统计 / Field Data Statistics ===")
        print(f"数据形状 / Data shape: {intensity_data.shape}")
        print(f"最大强度 / Maximum intensity: {np.max(intensity_data):.2e}")
        print(f"最小强度 / Minimum intensity: {np.min(intensity_data):.2e}")
        print(f"平均强度 / Mean intensity: {np.mean(intensity_data):.2e}")
        print(f"标准差 / Standard deviation: {np.std(intensity_data):.2e}")
        
        # 检查是否有多环结构
        center_value = intensity_data[center_y, center_x]
        max_value = np.max(intensity_data)
        print(f"\n中心强度 / Center intensity: {center_value:.2e}")
        print(f"中心/最大强度比 / Center/Max ratio: {center_value/max_value:.3f}")
        
        if center_value < 0.1 * max_value:
            print("检测到中心暗斑，符合LP71模式特征 / Central dark spot detected, consistent with LP71 mode")
        
        return intensity_data
        
    except Exception as e:
        print(f"加载CSV文件失败: {e}")
        return None

def visualize_3d_field(csv_filename):
    """
    创建3D可视化
    Create 3D visualization
    """
    try:
        intensity_data = np.loadtxt(csv_filename, delimiter=',')
        
        # 创建3D图
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建坐标网格
        x = np.arange(intensity_data.shape[1])
        y = np.arange(intensity_data.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # 3D表面图
        surf = ax.plot_surface(X, Y, intensity_data, cmap='hot', alpha=0.8)
        
        ax.set_title('LP71光场3D强度分布 / LP71 Field 3D Intensity Distribution')
        ax.set_xlabel('X 像素 / X Pixel')
        ax.set_ylabel('Y 像素 / Y Pixel')
        ax.set_zlabel('强度 / Intensity')
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.show()
        
    except Exception as e:
        print(f"3D可视化失败: {e}")

if __name__ == "__main__":
    # 可视化LP71 CSV数据
    csv_filename = "final_intensity_LP71.csv"
    
    print("开始可视化LP71光场数据...")
    intensity_data = visualize_csv_field(csv_filename)
    
    if intensity_data is not None:
        # 询问是否显示3D图
        print("\n是否显示3D可视化？(y/n)")
        choice = input().lower()
        if choice == 'y' or choice == 'yes':
            visualize_3d_field(csv_filename)
    
    print("\n可视化完成！")