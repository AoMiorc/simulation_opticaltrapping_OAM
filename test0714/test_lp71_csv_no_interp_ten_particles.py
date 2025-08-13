import numpy as np
import os

# 导入必要的模块
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simulation'))
from particle import ParticleFactory
from environment import Environment
from trap import OpticalTrap
from box import SimulationBox

def analyze_particle_motion(trajectory):
    """
    Analyze particle motion, calculate average velocity and angular velocity around axis
    """
    all_results = []
    
    for particle_idx in range(len(trajectory)):
        positions = trajectory[particle_idx]['position']
        times = trajectory[particle_idx]['time']
        
        # 计算瞬时速度
        velocities = []
        angular_velocities = []
        
        for i in range(1, len(positions)):
            dt = times[i] - times[i-1]
            
            # 计算瞬时速度
            velocity = (positions[i] - positions[i-1]) / dt
            speed = np.linalg.norm(velocity)
            velocities.append(speed)
            
            # 计算绕Z轴的角速度
            r1 = positions[i-1][:2]  # 取x,y坐标
            r2 = positions[i][:2]
            
            # 计算角度变化
            angle1 = np.arctan2(r1[1], r1[0])
            angle2 = np.arctan2(r2[1], r2[0])
            
            # 处理角度跳跃（-π到π的跳跃）
            dangle = angle2 - angle1
            if dangle > np.pi:
                dangle -= 2*np.pi
            elif dangle < -np.pi:
                dangle += 2*np.pi
                
            angular_velocity = dangle / dt
            angular_velocities.append(angular_velocity)
        
        # 计算平均值
        avg_speed = np.mean(velocities) if velocities else 0
        avg_angular_velocity = np.mean(angular_velocities) if angular_velocities else 0
        
        # 计算径向距离统计
        radial_distances = np.linalg.norm(positions[:, :2], axis=1)
        avg_radius = np.mean(radial_distances)
        max_radius = np.max(radial_distances)
        min_radius = np.min(radial_distances)
        
        # 计算轨道周期（如果有明显的周期性运动）
        if len(angular_velocities) > 0 and np.abs(avg_angular_velocity) > 1e-3:
            orbital_period = 2 * np.pi / np.abs(avg_angular_velocity)
        else:
            orbital_period = None
        
        particle_result = {
            'particle_id': particle_idx,
            'avg_speed': avg_speed,
            'avg_angular_velocity': avg_angular_velocity,
            'avg_radius': avg_radius,
            'max_radius': max_radius,
            'min_radius': min_radius,
            'orbital_period': orbital_period,
            'velocities': velocities,
            'angular_velocities': angular_velocities,
            'radial_distances': radial_distances
        }
        all_results.append(particle_result)
    
    return all_results

def test_lp71_with_csv_field_ten_particles():
    """Test ten particles motion using new CSV optical field data with new API"""
    
    print("Starting LP71 new CSV optical field test with 10 particles (New API)...")
    
    # 1. Create 10 particles with uniformly distributed initial positions
    particles = []
    num_particles = 10
    
    # 生成在0到3e-6之间均匀分布的初始位置
    x_positions = np.linspace(0, 1.2e-6, num_particles)
    
    for i in range(num_particles):
        particle = ParticleFactory.create_polystyrene_sphere(
            radius=50e-9,
            position=np.array([x_positions[i], 0.0, 0.0])  # 均匀分布的初始位置
        )
        particles.append(particle)
        print(f"Created particle {i+1}: radius={particle.radius*1e9:.1f}nm, position=({x_positions[i]*1e6:.2f}, 0.0, 0.0) μm")
    
    # 2. 创建环境
    environment = Environment(
        medium='liquid',
        T=273.0,
        eta=0.001
    )
    
    # 3. 创建光阱 - 移除kappa参数
    optical_trap = OpticalTrap(
        center=np.array([0.0, 0.0, 0.0]),
        wavelength=1064e-9,
        laser_power=1.5,
        w0=2.5e-6,
        l=-6
    )
    
    # 4. 设置网格范围 - 增加分辨率
    x_range = np.linspace(-6e-6, 6e-6, 300)  # 从60增加到300
    y_range = np.linspace(-6e-6, 6e-6, 300)  # 从60增加到300
    z_range = np.linspace(-3e-6, 3e-6, 150)  # 从30增加到150
    
    # 5. 使用新API直接设置CSV光场数据
    intensity_csv = os.path.join(os.path.dirname(__file__), "final_intensity_LP71_m6_10cm_5times_smaller.csv")
    phase_csv = os.path.join(os.path.dirname(__file__), "final_phase_LP71_m6_10cm_5times_smaller.csv")
    
    # 使用新的setup_csv_field方法
    success = optical_trap.setup_csv_field(intensity_csv, phase_csv, x_range, y_range, z_range)
    
    if not success:
        print("Unable to setup CSV optical field, program exits")
        return None, None
    
    print("New CSV optical field setup completed using new API")
    
    # 6. 创建模拟盒子
    sim_box = SimulationBox(
        particles=particles,  # 传入粒子列表
        environment=environment,
        optical_trap=optical_trap
    )
    
    # 设置仿真参数
    sim_box.timestep = 5e-4  # 50μs
    sim_box.time = 0.0
    
    # 初始化阻尼系数（为每个粒子）
    gamma_single = environment.get_drag_coefficient(particles[0])
    sim_box.gamma = np.array([
        [gamma_single, gamma_single, gamma_single] for _ in range(num_particles)
    ])
    
    # 7. 运行模拟
    print("Starting simulation...")
    duration = 1  # 1 second
    trajectory = sim_box.simulate(duration)
    
    # 8. 保存结果
    output_file = "particle_trajectory_lp71_csv_ten_particles.csv"
    sim_box.save_trajectory_to_csv(output_file)
    print(f"Trajectory data saved to: {output_file}")
    
    # 9. 详细运动分析
    motion_analyses = analyze_particle_motion(trajectory)
    
    print("\n=== Simulation Results with New CSV Optical Field (10 Particles) ===")
    
    for i, analysis in enumerate(motion_analyses):
        final_position = trajectory[i]['position'][-1]
        max_displacement = np.max(np.linalg.norm(trajectory[i]['position'], axis=1))
        
        print(f"\n--- Particle {i+1} ---")
        print(f"Initial position: ({x_positions[i]*1e6:.2f}, 0.0, 0.0) μm")
        print(f"Final position: ({final_position[0]*1e6:.2f}, {final_position[1]*1e6:.2f}, {final_position[2]*1e6:.2f}) μm")
        print(f"Maximum displacement: {max_displacement*1e6:.2f} μm")
        print(f"Average speed: {analysis['avg_speed']*1e6:.3f} μm/s")
        print(f"Average angular velocity: {analysis['avg_angular_velocity']:.3f} rad/s")
        print(f"Average orbital radius: {analysis['avg_radius']*1e6:.2f} μm")
        
        if analysis['orbital_period'] is not None:
            print(f"Estimated orbital period: {analysis['orbital_period']:.3f} s")
            print(f"Estimated orbital frequency: {1/analysis['orbital_period']:.3f} Hz")
        else:
            print("No obvious orbital motion detected")
    
    # 计算整体统计
    all_speeds = [analysis['avg_speed'] for analysis in motion_analyses]
    all_angular_velocities = [analysis['avg_angular_velocity'] for analysis in motion_analyses]
    all_radii = [analysis['avg_radius'] for analysis in motion_analyses]
    
    print("\n=== Overall Statistics ===")
    print(f"Average speed across all particles: {np.mean(all_speeds)*1e6:.3f} ± {np.std(all_speeds)*1e6:.3f} μm/s")
    print(f"Average angular velocity across all particles: {np.mean(all_angular_velocities):.3f} ± {np.std(all_angular_velocities):.3f} rad/s")
    print(f"Average orbital radius across all particles: {np.mean(all_radii)*1e6:.2f} ± {np.std(all_radii)*1e6:.2f} μm")
    
    return trajectory, sim_box

# Main program
if __name__ == "__main__":
    trajectory, sim_box = test_lp71_with_csv_field_ten_particles()
    
    if trajectory is not None:
        # Visualization
        try:
            from visualizer import TrajectoryVisualizer
            import matplotlib.pyplot as plt
            
            visualizer = TrajectoryVisualizer("particle_trajectory_lp71_csv_ten_particles.csv")
            
            # 重新创建光阱用于可视化 - 移除kappa参数
            optical_trap = OpticalTrap(
                center=np.array([0.0, 0.0, 0.0]),
                wavelength=1064e-9,
                laser_power=0.15,
                w0=2.5e-6,
                l=-6
            )
            
            # 重新设置CSV光场 - 使用新API
            x_range = np.linspace(-6e-6, 6e-6, 200)
            y_range = np.linspace(-6e-6, 6e-6, 200)
            z_range = np.linspace(-3e-6, 3e-6, 100)
            
            intensity_csv = os.path.join(os.path.dirname(__file__), "final_intensity_LP71_m6_10cm_5times_smaller.csv")
            phase_csv = os.path.join(os.path.dirname(__file__), "final_phase_LP71_m6_10cm_5times_smaller.csv")
            
            # 使用新API设置光场
            success = optical_trap.setup_csv_field(intensity_csv, phase_csv, x_range, y_range, z_range)
            
            if success:
                # 绘制所有粒子的轨迹
                visualizer.plot_2d_trajectory_with_point_field('xy', 
                                                              optical_trap=optical_trap, 
                                                              field_alpha=0.6)
                
                # 显示图形
                plt.show()
            
        except ImportError as e:
            print(f"Visualization module import failed: {e}")
            print("Skipping visualization step")
        
        print("\nLP71 new CSV optical field test with 10 particles completed!")
