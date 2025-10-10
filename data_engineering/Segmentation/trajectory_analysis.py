# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import argparse
# import os
# import re
# import seaborn as sns
# import glob

# def parse_time_from_filename(filename):
#     """
#     Parses a timestamp like '...00d00h30m.tif' into a pandas Timedelta object.
#     Returns None if the pattern is not found.
#     """
#     match = re.search(r'(\d+)d(\d+)h(\d+)m', filename)
#     if match:
#         days, hours, minutes = map(int, match.groups())
#         return pd.to_timedelta(f'{days} days {hours} hours {minutes} minutes')
#     return None

# def analyze_trajectories(args):
#     """
#     Loads trajectory data, automatically calculates dt from filenames,
#     calculates speeds and total distances, and generates specified plots.
#     """
#     # If 'all' is requested, set the plots list to include all plot types
#     if 'all' in args.plots:
#         args.plots = ['speed_histogram', 'timeseries', 'distance_histogram']
        
#     try:
#         df_traj = pd.read_csv(args.infile)
#     except FileNotFoundError:
#         print(f"Error: Input file not found at '{args.infile}'")
#         return

#     if not os.path.isdir(args.features_folder):
#         print(f"Error: The specified features folder was not found at '{args.features_folder}'")
#         return

#     filenames = sorted(glob.glob(os.path.join(args.features_folder, '*.tif')))
#     time_data = []
#     for i, f in enumerate(filenames):
#         timestamp = parse_time_from_filename(os.path.basename(f))
#         if timestamp is not None:
#             time_data.append({'frame': i, 'timestamp': timestamp})
    
#     if not time_data:
#         print("Error: Could not parse timestamps from any filenames in the features folder. Aborting.")
#         return

#     df_time = pd.DataFrame(time_data)
#     df = pd.merge(df_traj, df_time, on='frame')
#     df = df.sort_values(by=['particle', 'timestamp']).reset_index(drop=True)

#     df['dx'] = df.groupby('particle')['x'].diff()
#     df['dy'] = df.groupby('particle')['y'].diff()
#     df['dt'] = df.groupby('particle')['timestamp'].diff()
#     df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2) * args.pixelsize

#     if args.time_base == 'minutes':
#         df['dt_val'] = df['dt'].dt.total_seconds() / 60.0
#         time_unit_label = 'min'
#     elif args.time_base == 'hours':
#         df['dt_val'] = df['dt'].dt.total_seconds() / 3600.0
#         time_unit_label = 'hr'
#     else: # seconds
#         df['dt_val'] = df['dt'].dt.total_seconds()
#         time_unit_label = 'sec'

#     df['speed'] = df['distance'] / df['dt_val']
#     df['time_val'] = df['timestamp'].dt.total_seconds() / (60.0 if args.time_base == 'minutes' else 3600.0 if args.time_base == 'hours' else 1.0)

#     df_analysis = df.dropna(subset=['speed']).copy()
#     os.makedirs(args.output_folder, exist_ok=True)

#     if 'speed_histogram' in args.plots:
#         print("\nSpeed Statistics (per step):")
#         print(df_analysis['speed'].describe())
        
#         plt.figure(figsize=(8, 6))
#         plt.hist(df_analysis['speed'], bins='auto', color='skyblue', ec='black')
#         plt.title('Distribution of Cell Speeds')
#         plt.xlabel(f"Speed ({args.unit_name}/{time_unit_label})")
#         plt.ylabel('Frequency (Counts)')
#         plt.grid(axis='y', alpha=0.75)
        
#         output_path = os.path.join(args.output_folder, "speed_histogram.png")
#         plt.savefig(output_path, dpi=150)
#         plt.close()
#         print(f"Speed histogram saved to: {output_path}")

#     if 'timeseries' in args.plots:
#         plt.figure(figsize=(12, 7))
#         sns.lineplot(data=df_analysis, x='time_val', y='speed', hue='particle', legend=None)
#         plt.title('Cell Speed Over Time')
#         plt.xlabel(f"Time ({time_unit_label})")
#         plt.ylabel(f"Speed ({args.unit_name}/{time_unit_label})")
#         plt.grid(alpha=0.5)

#         output_path = os.path.join(args.output_folder, "speed_timeseries.png")
#         plt.savefig(output_path, dpi=150)
#         plt.close()
#         print(f"Time series plot saved to: {output_path}")
        
#     if 'distance_histogram' in args.plots:
#         total_distances = df.groupby('particle')['distance'].sum()
        
#         print("\nTotal Distance Statistics (per trajectory):")
#         print(total_distances.describe())
        
#         plt.figure(figsize=(8, 6))
#         plt.hist(total_distances, bins='auto', color='coral', ec='black')
#         plt.title('Distribution of Total Distance Traveled per Trajectory')
#         plt.xlabel(f"Total Distance ({args.unit_name})")
#         plt.ylabel('Frequency (Counts of Trajectories)')
#         plt.grid(axis='y', alpha=0.75)
        
#         output_path = os.path.join(args.output_folder, "distance_histogram.png")
#         plt.savefig(output_path, dpi=150)
#         plt.close()
#         print(f"Total distance histogram saved to: {output_path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Analyze cell speeds and distances from a trackpy trajectories file.")
    
#     parser.add_argument("--infile", required=True, help="Path to the input trajectories.csv file.")
#     parser.add_argument("--features_folder", required=True, help="Path to the 'trackpy_features' folder for timestamps.")
#     parser.add_argument("--output_folder", default=".", help="Directory to save the output plots.")
    
#     parser.add_argument("--pixelsize", type=float, default=1.0, help="Physical size of one pixel (e.g., in um, mm).")
#     parser.add_argument("--unit-name", default="pixel", help="Name of the spatial unit (e.g., 'um', 'mm').")
#     parser.add_argument("--time-base", choices=['seconds', 'minutes', 'hours'], default='minutes', help="The time unit for speed calculation and plot axes.")
    
#     parser.add_argument("--plots", nargs='+', choices=['speed_histogram', 'timeseries', 'distance_histogram', 'all'], default=['all'], help="Specify which plots to generate.")
    
#     args = parser.parse_args()
    
#     analyze_trajectories(args)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re
import seaborn as sns
import glob
import trackpy as tp

def parse_time_from_filename(filename):
    """
    Parses a timestamp like '...00d00h30m.tif' into a pandas Timedelta object.
    Returns None if the pattern is not found.
    """
    match = re.search(r'(\d+)d(\d+)h(\d+)m', filename)
    if match:
        days, hours, minutes = map(int, match.groups())
        return pd.to_timedelta(f'{days} days {hours} hours {minutes} minutes')
    return None

def analyze_trajectories(args):
    """
    Loads trajectory data, automatically calculates dt from filenames,
    calculates speeds, distances, and diffusion constants, and generates specified plots.
    """
    # If 'all' is requested, set the plots list to include all plot types
    if 'all' in args.plots:
        args.plots = ['speed_histogram', 'timeseries', 'distance_histogram', 'diffusion_histogram']
        
    try:
        df_traj = pd.read_csv(args.infile)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.infile}'")
        return

    if not os.path.isdir(args.features_folder):
        print(f"Error: The specified features folder was not found at '{args.features_folder}'")
        return

    filenames = sorted(glob.glob(os.path.join(args.features_folder, '*.tif')))
    time_data = []
    for i, f in enumerate(filenames):
        timestamp = parse_time_from_filename(os.path.basename(f))
        if timestamp is not None:
            time_data.append({'frame': i, 'timestamp': timestamp})
    
    if not time_data:
        print("Error: Could not parse timestamps from any filenames in the features folder. Aborting.")
        return

    df_time = pd.DataFrame(time_data)
    df = pd.merge(df_traj, df_time, on='frame')
    df = df.sort_values(by=['particle', 'timestamp']).reset_index(drop=True)

    # Convert coordinates to physical units early
    df['x_um'] = df['x'] * args.pixelsize
    df['y_um'] = df['y'] * args.pixelsize

    # Convert timestamp to a numerical value for calculations and plotting
    if args.time_base == 'minutes':
        time_divisor = 60.0
        time_unit_label = 'min'
    elif args.time_base == 'hours':
        time_divisor = 3600.0
        time_unit_label = 'hr'
    else: # seconds
        time_divisor = 1.0
        time_unit_label = 'sec'
    df['time_val'] = df['timestamp'].dt.total_seconds() / time_divisor

    # --- Calculations for Speed and Distance ---
    df['dx'] = df.groupby('particle')['x_um'].diff()
    df['dy'] = df.groupby('particle')['y_um'].diff()
    df['dt'] = df.groupby('particle')['time_val'].diff()
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['speed'] = df['distance'] / df['dt']

    df_analysis = df.dropna(subset=['speed']).copy()
    os.makedirs(args.output_folder, exist_ok=True)

    # Plotting Section 
    
    if 'speed_histogram' in args.plots:
        print("\nSpeed Statistics (per step):")
        print(df_analysis['speed'].describe())
        plt.figure(figsize=(8, 6))
        plt.hist(df_analysis['speed'], bins='auto', color='skyblue', ec='black')
        plt.title('Distribution of Cell Speeds')
        plt.xlabel(f"Speed ({args.unit_name}/{time_unit_label})")
        plt.ylabel('Frequency (Counts)')
        plt.grid(axis='y', alpha=0.75)
        output_path = os.path.join(args.output_folder, "speed_histogram.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Speed histogram saved to: {output_path}")

    if 'timeseries' in args.plots:
        plt.figure(figsize=(12, 7))
        sns.lineplot(data=df_analysis, x='time_val', y='speed', hue='particle', legend=None)
        plt.title('Cell Speed Over Time')
        plt.xlabel(f"Time ({time_unit_label})")
        plt.ylabel(f"Speed ({args.unit_name}/{time_unit_label})")
        plt.grid(alpha=0.5)
        output_path = os.path.join(args.output_folder, "speed_timeseries.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Time series plot saved to: {output_path}")
        
    if 'distance_histogram' in args.plots:
        total_distances = df.groupby('particle')['distance'].sum()
        print("\nTotal Distance Statistics (per trajectory):")
        print(total_distances.describe())
        plt.figure(figsize=(8, 6))
        plt.hist(total_distances, bins='auto', color='coral', ec='black')
        plt.title('Distribution of Total Distance Traveled per Trajectory')
        plt.xlabel(f"Total Distance ({args.unit_name})")
        plt.ylabel('Frequency (Counts of Trajectories)')
        plt.grid(axis='y', alpha=0.75)
        output_path = os.path.join(args.output_folder, "distance_histogram.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Total distance histogram saved to: {output_path}")

    if 'diffusion_histogram' in args.plots:
        # Use trackpy's imsd. The result is a WIDE format DataFrame
        # where columns are particle IDs.
        imsd_wide = tp.imsd(df, mpp=1, fps=1, max_lagtime=len(df_time)-1)
        
        diffusion_constants = []
        
        # The first column is the lag time, the rest are particle MSDs.
        lag_times_frames = imsd_wide.iloc[:, 0]
        particle_columns = imsd_wide.columns[1:]
        
        # Iterate through each particle column
        for particle_id in particle_columns:
            msd_data_pixels = imsd_wide[particle_id]
            
            # Combine into a temporary DataFrame and drop missing values
            temp_df = pd.DataFrame({
                'lagtime': lag_times_frames,
                'msd': msd_data_pixels
            }).dropna()

            fit_data = temp_df.head(args.msd_fit_points)
            
            if len(fit_data) < 2:
                continue

            avg_dt = df['dt'].mean()
            lag_time_real = fit_data['lagtime'] * avg_dt
            msd_real = fit_data['msd'] * (args.pixelsize**2)

            slope, intercept = np.polyfit(lag_time_real, msd_real, 1)
            
            D = slope / 4
            diffusion_constants.append(D)
        
        diffusion_constants = pd.Series(diffusion_constants)
        print("\nDiffusion Constant D Statistics (per trajectory):")
        print(diffusion_constants.describe())

        plt.figure(figsize=(8, 6))
        plt.hist(diffusion_constants.dropna().values, bins='auto', color='mediumseagreen', ec='black')
        
        plt.title('Distribution of Diffusion Constants (D)')
        plt.xlabel(f"Diffusion Constant D ({args.unit_name}²/{time_unit_label})")
        plt.ylabel('Frequency (Counts of Trajectories)')
        plt.grid(axis='y', alpha=0.75)
        
        output_path = os.path.join(args.output_folder, "diffusion_histogram.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Diffusion constant histogram saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze cell speeds, distances, and diffusion from a trackpy trajectories file.")
    
    parser.add_argument("--infile", required=True, help="Path to the input trajectories.csv file.")
    parser.add_argument("--features_folder", required=True, help="Path to the 'trackpy_features' folder for timestamps.")
    parser.add_argument("--output_folder", default=".", help="Directory to save the output plots.")
    
    parser.add_argument("--pixelsize", type=float, default=1.0, help="Physical size of one pixel (e.g., in um, mm).")
    parser.add_argument("--unit-name", default="pixel", help="Name of the spatial unit (e.g., 'um', 'mm').")
    parser.add_argument("--time-base", choices=['seconds', 'minutes', 'hours'], default='minutes', help="The time unit for speed, distance, and diffusion calculations.")
    
    parser.add_argument("--plots", nargs='+', choices=['speed_histogram', 'timeseries', 'distance_histogram', 'diffusion_histogram', 'all'], default=['all'], help="Specify which plots to generate.")
    parser.add_argument("--msd_fit_points", type=int, default=10, help="Number of initial MSD points to use for the linear fit to find D.")
    
    args = parser.parse_args()
    
    analyze_trajectories(args)
