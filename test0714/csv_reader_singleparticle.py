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
        # 1. Create particles
        particle = ParticleFactory.create_polystyrene_sphere(
            radius=50e-9,
            position=np.array([0.1e-6, 0.0, 0.0])
        )
        print(f"Created particle: radius={particle.radius*1e9:.1f}nm")
        
        # 2. Create environment
        environment = Environment(
            medium='liquid',
            T=197.0,
            eta=0.001
        )
        
        # 3. Create a light trap
        optical_trap = OpticalTrap(
            center=np.array([0.0, 0.0, 0.0]),
            wavelength=1064e-9,
            laser_power=0.15,
            w0=2.5e-6,
            l=np.nan
        )
        
        # 4. Set up CSV optical field
        intensity_csv = os.path.join(os.path.dirname(__file__), "final_intensity_LP71_m6_0cm.csv")
        phase_csv = os.path.join(os.path.dirname(__file__), "final_phase_LP71_m6_0cm.csv")
        
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
            particles=particle,
            environment=environment,
            optical_trap=optical_trap
        )
        
        # Set up simulation parameters  
        sim_box.timestep = 5e-4
        sim_box.time = 0.0
        sim_box.gamma = np.array([
            environment.get_drag_coefficient(particle),
            environment.get_drag_coefficient(particle),
            environment.get_drag_coefficient(particle)
        ])
        
        # 6. Run the simulation
        print("Starting simulation...")
        duration = 0.01
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
        visualizer.analyze_and_visualize_default(sim_box=sim_box, show_plots=True)
        
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()