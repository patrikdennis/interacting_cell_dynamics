import trackpy as tp
import pandas as pd
import pims
import matplotlib.pyplot as plt
import argparse
import os

def run_particle_tracking(args):
    """
    Performs particle tracking on a sequence of feature images using trackpy.
    All parameters are passed via the 'args' object from argparse.
    """
    print("Starting Trackpy analysis...")

    # Load the image sequence
    try:
        frames = pims.ImageSequence(os.path.join(args.input_folder, '*.tif'))
        first_frame = frames[0]
    except (FileNotFoundError, IndexError):
        print(f"Error: Could not find or load images from '{args.input_folder}'.")
        print("Please ensure the directory exists and contains .tif images.")
        return
    
    print(f"Loaded {len(frames)} frames from '{args.input_folder}'.")

    # Locate features in all frames
    print(f"Finding features with diameter={args.diameter} and minmass={args.minmass}...")
    features = tp.batch(frames, diameter=args.diameter, minmass=args.minmass, invert=False)
    print(f"Found a total of {len(features)} features across all frames.")

    # Link features into trajectories
    print(f"Linking trajectories with search_range={args.max_displacement} and memory={args.memory}...")
    trajectories = tp.link(features, search_range=args.max_displacement, memory=args.memory)

    # Filter out short, spurious trajectories
    print(f"Filtering out trajectories shorter than {args.min_length} frames...")
    filtered_trajectories = tp.filter_stubs(trajectories, threshold=args.min_length)
    print(f"Kept {filtered_trajectories['particle'].nunique()} trajectories after filtering.")

    # Save the results
    output_csv_path = os.path.join(args.output_folder, "trajectories.csv")
    filtered_trajectories.to_csv(output_csv_path, index=False)
    print(f"\nTrajectory data saved to: {output_csv_path}")

    # Create and save a plot of the trajectories
    output_plot_path = os.path.join(args.output_folder, "trajectory_plot.png")
    plt.figure(figsize=(8, 8))
    plt.imshow(first_frame, cmap='gray')
    tp.plot_traj(filtered_trajectories, ax=plt.gca())
    plt.title("Detected Cell Trajectories")
    plt.savefig(output_plot_path, dpi=150)
    plt.close()
    print(f"Trajectory plot saved to: {output_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Trackpy on binary feature images.")
    
    # I/O Arguments
    parser.add_argument("--input-folder", required=True, help="Folder containing the binary feature images.")
    parser.add_argument("--output-folder", required=True, help="Folder to save tracking data (CSV) and plots.")
    
    # Trackpy Feature Finding Arguments
    parser.add_argument("--diameter", type=int, default=5, help="Feature diameter (must be an odd integer). For 5x5 squares, a value of 5 is ideal.")
    parser.add_argument("--minmass", type=float, default=100, help="Minimum integrated brightness of a feature.")
    
    # Trackpy Linking Arguments
    parser.add_argument("--max-displacement", type=int, default=20, help="Maximum distance a feature can move between frames.")
    parser.add_argument("--memory", type=int, default=3, help="Number of frames a feature can be lost and remembered.")
    
    # Trackpy Filtering Arguments
    parser.add_argument("--min-length", type=int, default=4, help="Minimum length of a trajectory to be kept (stub filter length).")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    run_particle_tracking(args)