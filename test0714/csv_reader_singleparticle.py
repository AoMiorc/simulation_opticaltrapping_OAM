import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simulation'))
from particle import ParticleFactory
from environment import Environment
from trap import OpticalTrap
from box import SimulationBox
from visualizer import TrajectoryVisualizer

def main():
    print("Starting LP71 CSV optical field test with enhanced features...")
    
    try:
        # 1. Create particles (5 particles with specific positions)
        particles = []
        # 直接定义每个粒子的位置 (x, y, z) 单位：米
        particle_positions = [
            [0.0e-6, 0.0, 0.0],      # 粒子1：原点
            [0.4e-6, 0.0, 0.0],    # 粒子2
            [0.8e-6, 0.0, 0.0],     # 粒子3
            [1.2e-6, 0.0, 0.0],    # 粒子4
            [1.6e-6, 0.0, 0.0]       # 粒子5：最外侧
        ]
        
        for i, pos in enumerate(particle_positions):
            particle = ParticleFactory.create_polystyrene_sphere(
                radius=0.5e-7,
                position=np.array(pos)
            )
            particles.append(particle)
            print(f"Created particle {i+1}: radius={particle.radius*1e9:.1f}nm, position=({pos[0]*1e6:.3f}μm, {pos[1]*1e6:.3f}μm, {pos[2]*1e6:.3f}μm)")
        
        print(f"Total particles created: {len(particles)}")
        
        # 2. Create environment
        environment = Environment(
            medium='liquid',
            T=0,
            eta=0.001
        )
        
        # 3. Create a light trap
        optical_trap = OpticalTrap(
            center=np.array([0.0, 0.0, 0.0]),
            wavelength=1064e-9,
            laser_power=1.5,
            w0=2.5e-6,
            l=np.nan
        )
        
        # 4. Set up CSV optical field
        intensity_csv = os.path.join(os.path.dirname(__file__), "final_intensity_LP01_m6_0cm.csv")
        phase_csv = os.path.join(os.path.dirname(__file__), "final_phase_LP01_m6_0cm.csv")
        
        x_range = np.linspace(-6e-6, 6e-6, 300)
        y_range = np.linspace(-6e-6, 6e-6, 300)
        z_range = np.linspace(-3e-6, 3e-6, 150)
        
        success = optical_trap.setup_csv_field(
            intensity_csv=intensity_csv,
            phase_csv=phase_csv,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range
        )
        
        if not success:
            print("CSV optical field setup failed")
            return
            
        print("CSV optical field setup completed")
        
        # 5. Create a simulation box
        sim_box = SimulationBox(
            particles=particles,  # 传入粒子列表
            environment=environment,
            optical_trap=optical_trap
        )
        
        # Set up simulation parameters  
        sim_box.timestep = 1e-8
        sim_box.time = 0.0
        # 为每个粒子设置阻力系数
        gamma_list = []
        for particle in particles:
            gamma_list.extend([
                environment.get_drag_coefficient(particle),
                environment.get_drag_coefficient(particle),
                environment.get_drag_coefficient(particle)
            ])
        sim_box.gamma = np.array(gamma_list)
        
        # 6. Run the simulation
        print("Starting simulation...")
        duration = 0.5e-3
        trajectory = sim_box.simulate(duration)
        
        # 7. Save the trajectory data
        output_file = "particle_trajectory_lp71_csv_new.csv"
        sim_box.save_trajectory_to_csv(output_file)
        print(f"Trajectory data saved to: {output_file}")
        
        # 8. Use default configuration for complete analysis and visualization  
        print("\n=== Starting Complete Analysis and Visualization ===")
        visualizer = TrajectoryVisualizer(output_file)
        visualizer.load_from_box(sim_box, csv_file=output_file)
        
        # Use the new default configuration method
        # 传入optical_trap参数以显示强度场背景
        visualizer.analyze_and_visualize_default(sim_box=sim_box, show_plots=True)
        visualizer.create_trajectory_video(
            output_filename='m6_10cm_5times_NEW.mp4',
            duration_sec=6, 
            optical_trap=sim_box.optical_trap,  # 关键参数
            plane='xy'  # 投影平面
        )
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
