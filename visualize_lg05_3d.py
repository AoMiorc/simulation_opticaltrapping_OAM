import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加simulation模块到路径 / Add simulation module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

from visualizer import TrajectoryVisualizer

def visualize_lg05_3d_trajectory():
    """可视化LG05测试数据的3D轨迹 / Visualize 3D trajectory of LG05 test data"""
    
    print("开始LG05轨迹3D可视化... / Starting LG05 trajectory 3D visualization...")
    
    # 检查CSV文件是否存在 / Check if CSV file exists
    csv_file = "particle_trajectory_lg05_test.csv"
    if not os.path.exists(csv_file):
        print(f"错误: 找不到文件 {csv_file} / Error: Cannot find file {csv_file}")
        print("请确保该文件存在于当前目录中 / Please ensure the file exists in current directory")
        return
    
    print(f"加载轨迹数据: {csv_file} / Loading trajectory data: {csv_file}")
    
    # 创建可视化器并加载数据 / Create visualizer and load data
    try:
        visualizer = TrajectoryVisualizer(csv_file)
        print("数据加载成功! / Data loaded successfully!")
        
        # 显示数据基本信息 / Show basic data information
        if visualizer.particles_data:
            particle_ids = list(visualizer.particles_data.keys())
            print(f"发现 {len(particle_ids)} 个粒子的轨迹数据 / Found trajectory data for {len(particle_ids)} particle(s)")
            
            for particle_id in particle_ids:
                data = visualizer.particles_data[particle_id]
                num_points = len(data)
                time_span = data['Time (s)'].iloc[-1] - data['Time (s)'].iloc[0]
                
                # 计算轨迹统计信息 / Calculate trajectory statistics
                x_range = data['X (m)'].max() - data['X (m)'].min()
                y_range = data['Y (m)'].max() - data['Y (m)'].min()
                z_range = data['Z (m)'].max() - data['Z (m)'].min()
                
                print(f"\n粒子 {particle_id} 统计信息: / Particle {particle_id} statistics:")
                print(f"  数据点数: {num_points} / Data points: {num_points}")
                print(f"  时间跨度: {time_span*1000:.2f} ms / Time span: {time_span*1000:.2f} ms")
                print(f"  X范围: {x_range*1e6:.2f} μm / X range: {x_range*1e6:.2f} μm")
                print(f"  Y范围: {y_range*1e6:.2f} μm / Y range: {y_range*1e6:.2f} μm")
                print(f"  Z范围: {z_range*1e6:.2f} μm / Z range: {z_range*1e6:.2f} μm")
                
                # 计算最大位移 / Calculate maximum displacement
                positions = np.column_stack([data['X (m)'], data['Y (m)'], data['Z (m)']])
                displacements = np.linalg.norm(positions, axis=1)
                max_displacement = np.max(displacements)
                print(f"  最大位移: {max_displacement*1e6:.2f} μm / Maximum displacement: {max_displacement*1e6:.2f} μm")
        
    except Exception as e:
        print(f"加载数据时出错: {e} / Error loading data: {e}")
        return
    
    print("\n开始绘制3D可视化图形... / Starting to plot 3D visualization...")
    
    # 1. 绘制主要的3D轨迹图 / Plot main 3D trajectory
    print("1. 绘制3D轨迹图... / 1. Plotting 3D trajectory...")
    try:
        visualizer.plot_3d_trajectory(figsize=(14, 10))
        plt.suptitle('LG05光束中粒子的3D轨迹 / 3D Trajectory of Particle in LG05 Beam', fontsize=16)
        plt.show()
    except Exception as e:
        print(f"绘制3D轨迹时出错: {e} / Error plotting 3D trajectory: {e}")
    
    # 2. 绘制XY平面投影 / Plot XY plane projection
    print("2. 绘制XY平面投影... / 2. Plotting XY plane projection...")
    try:
        visualizer.plot_2d_trajectory('xy', figsize=(10, 8))
        plt.suptitle('LG05光束中粒子轨迹 - XY平面投影 / Particle Trajectory in LG05 Beam - XY Projection', fontsize=14)
        plt.show()
    except Exception as e:
        print(f"绘制XY投影时出错: {e} / Error plotting XY projection: {e}")
    
    # 3. 绘制XZ平面投影 / Plot XZ plane projection
    print("3. 绘制XZ平面投影... / 3. Plotting XZ plane projection...")
    try:
        visualizer.plot_2d_trajectory('xz', figsize=(10, 8))
        plt.suptitle('LG05光束中粒子轨迹 - XZ平面投影 / Particle Trajectory in LG05 Beam - XZ Projection', fontsize=14)
        plt.show()
    except Exception as e:
        print(f"绘制XZ投影时出错: {e} / Error plotting XZ projection: {e}")
    
    # 4. 绘制YZ平面投影 / Plot YZ plane projection
    print("4. 绘制YZ平面投影... / 4. Plotting YZ plane projection...")
    try:
        visualizer.plot_2d_trajectory('yz', figsize=(10, 8))
        plt.suptitle('LG05光束中粒子轨迹 - YZ平面投影 / Particle Trajectory in LG05 Beam - YZ Projection', fontsize=14)
        plt.show()
    except Exception as e:
        print(f"绘制YZ投影时出错: {e} / Error plotting YZ projection: {e}")
    
    # 5. 绘制速度大小随时间变化 / Plot velocity magnitude vs time
    print("5. 绘制速度分析图... / 5. Plotting velocity analysis...")
    try:
        visualizer.plot_velocity_magnitude(figsize=(12, 6))
        plt.suptitle('LG05光束中粒子的速度和角速度分析 / Velocity and Angular Velocity Analysis in LG05 Beam', fontsize=14)
        plt.show()
    except Exception as e:
        print(f"绘制速度分析时出错: {e} / Error plotting velocity analysis: {e}")
    
    print("\n=== LG05轨迹3D可视化完成! / LG05 Trajectory 3D Visualization Completed! ===")
    print("已生成以下图形: / Generated the following plots:")
    print("1. 完整3D轨迹图 / Complete 3D trajectory plot")
    print("2. XY平面投影 / XY plane projection")
    print("3. XZ平面投影 / XZ plane projection")
    print("4. YZ平面投影 / YZ plane projection")
    print("5. 速度和角速度分析 / Velocity and angular velocity analysis")
    
    return visualizer

def interactive_3d_view():
    """交互式3D视图 / Interactive 3D view"""
    print("\n启动交互式3D视图... / Starting interactive 3D view...")
    
    csv_file = "particle_trajectory_lg05_test.csv"
    if not os.path.exists(csv_file):
        print(f"错误: 找不到文件 {csv_file} / Error: Cannot find file {csv_file}")
        return
    
    try:
        visualizer = TrajectoryVisualizer(csv_file)
        
        # 创建交互式3D图 / Create interactive 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 获取数据 / Get data
        particle_ids = list(visualizer.particles_data.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(particle_ids)))
        
        for i, particle_id in enumerate(particle_ids):
            data = visualizer.particles_data[particle_id]
            
            # 绘制轨迹线 / Plot trajectory line
            ax.plot(data['X (m)'], data['Y (m)'], data['Z (m)'], 
                   color=colors[i], linewidth=2, alpha=0.8, 
                   label=f'粒子 {particle_id} / Particle {particle_id}')
            
            # 标记起点和终点 / Mark start and end points
            ax.scatter(data['X (m)'].iloc[0], data['Y (m)'].iloc[0], 
                      data['Z (m)'].iloc[0], color=colors[i], s=100, 
                      marker='o', edgecolors='white', linewidth=2, 
                      label=f'起点 {particle_id} / Start {particle_id}')
            ax.scatter(data['X (m)'].iloc[-1], data['Y (m)'].iloc[-1], 
                      data['Z (m)'].iloc[-1], color=colors[i], s=100, 
                      marker='s', edgecolors='white', linewidth=2,
                      label=f'终点 {particle_id} / End {particle_id}')
        
        # 设置坐标轴标签 / Set axis labels
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('LG05光束中粒子的交互式3D轨迹\nInteractive 3D Trajectory in LG05 Beam', fontsize=14)
        
        # 添加图例 / Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 设置相等的坐标轴比例 / Set equal axis scaling
        # ax.set_box_aspect([1,1,1])
        
        plt.tight_layout()
        plt.show()
        
        print("交互式3D视图已显示，你可以用鼠标旋转和缩放 / Interactive 3D view displayed, you can rotate and zoom with mouse")
        
    except Exception as e:
        print(f"创建交互式视图时出错: {e} / Error creating interactive view: {e}")

if __name__ == "__main__":
    print("=== LG05轨迹数据3D可视化工具 / LG05 Trajectory Data 3D Visualization Tool ===")
    print("\n选择模式: / Choose mode:")
    print("1. 完整可视化分析 (推荐) / Complete visualization analysis (Recommended)")
    print("2. 仅交互式3D视图 / Interactive 3D view only")
    
    try:
        choice = input("\n请输入选择 (1 或 2): / Please enter choice (1 or 2): ").strip()
        
        if choice == "1" or choice == "":
            # 运行完整的可视化分析 / Run complete visualization analysis
            visualizer = visualize_lg05_3d_trajectory()
            
            # 询问是否要显示交互式视图 / Ask if want to show interactive view
            show_interactive = input("\n是否显示交互式3D视图? (y/n): / Show interactive 3D view? (y/n): ").strip().lower()
            if show_interactive in ['y', 'yes', '是', '']:
                interactive_3d_view()
                
        elif choice == "2":
            # 仅显示交互式3D视图 / Show interactive 3D view only
            interactive_3d_view()
            
        else:
            print("无效选择，运行默认模式 / Invalid choice, running default mode")
            visualize_lg05_3d_trajectory()
            
    except KeyboardInterrupt:
        print("\n\n用户取消操作 / User cancelled operation")
    except Exception as e:
        print(f"\n运行时出错: {e} / Runtime error: {e}")
        # 尝试运行基本可视化 / Try to run basic visualization
        print("尝试运行基本可视化... / Trying to run basic visualization...")
        try:
            visualize_lg05_3d_trajectory()
        except:
            print("基本可视化也失败了 / Basic visualization also failed")