import numpy as np
import matplotlib.pyplot as plt

def visualize_csv(intensity_file, phase_file):
    try:
        intensity_data = np.loadtxt(intensity_file, delimiter=',')
        phase_data = np.loadtxt(phase_file, delimiter=',')
        
        if intensity_data.shape != phase_data.shape:
            print("警告：文件尺寸不匹配，无法正确可视化。")
            return
        
        # 假设数据是2D网格，x和y坐标从0到shape
        extent = [0, intensity_data.shape[1], 0, intensity_data.shape[0]]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(intensity_data, cmap='hot', extent=extent, origin='lower')
        plt.colorbar()
        plt.title('Intensity Map')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        plt.subplot(1, 2, 2)
        plt.imshow(phase_data, cmap='hsv', extent=extent, origin='lower')
        plt.colorbar()
        plt.title('Phase Map')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"错误：{e}")

# 使用示例
intensity_path = 'c:\\Users\\mioer\\Optical_image_python\\test0714\\final_intensity_LP71_m6_10cm.csv'
phase_path = 'c:\\Users\\mioer\\Optical_image_python\\test0714\\final_phase_LP71_m6_10cm.csv'
visualize_csv(intensity_path, phase_path)