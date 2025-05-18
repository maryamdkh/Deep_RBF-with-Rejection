import os
import pickle
import numpy as np
import pandas as pd
from sktime.transformations.panel.rocket import Rocket
from pycatch22 import catch22_all
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from multiprocessing import Pool
from functools import partial
import json

class TimeSeriesFeatureProcessor:
    def __init__(self, method='rocket', output_dir='./features', 
                 rocket_n_kernels=10_000, n_jobs=-1):
        """
        Args:
            method: Feature extraction method ('rocket', 'handcrafted', 'tsfresh', 'catch22', or 'raw')
            output_dir: Directory to save feature files
            rocket_n_kernels: Number of kernels for ROCKET method
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.method = method
        self.output_dir = output_dir
        self.rocket_n_kernels = rocket_n_kernels
        self.n_jobs = n_jobs
        self.rocket = None
        
        os.makedirs(output_dir, exist_ok=True)

    def _load_sample(self, file_path):
        """Load and preprocess a single sample with time component"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Filter lines based on length threshold
        valid_lines = [line for line in data['lines'] if len(line['points']) >= 10]
        
        if not valid_lines:
            return None
        
        # Convert to numpy array [n_lines, n_points, 3] where last dim is [time, x, y]
        sample_data = []
        for line in valid_lines:
            points = np.array([[p['x'], p['y']] for p in line['points']])
            times = np.array(line['times']).reshape(-1, 1)  # Reshape to column vector
            # Combine time with coordinates [time, x, y]
            line_data = np.concatenate([times, points], axis=1)
            sample_data.append(line_data)
        
        return np.stack(sample_data)
    
    def _extract_features_single(self, sample_data):
        """Extract features for a single sample with time component"""
        if sample_data is None:
            return None
            
        if self.method == 'raw':
            return sample_data  # Return raw time series with time component
            
        elif self.method == 'rocket':
            # Reshape for ROCKET: [1, n_lines*3, n_timesteps] (time, x, y for each line)
            n_lines = sample_data.shape[0]
            rocket_sample = np.zeros((1, n_lines*3, sample_data.shape[1]))
            
            for i in range(n_lines):
                rocket_sample[0, 3*i] = sample_data[i, :, 0]  # time
                rocket_sample[0, 3*i+1] = sample_data[i, :, 1]  # x coordinates
                rocket_sample[0, 3*i+2] = sample_data[i, :, 2]  # y coordinates
            
            if self.rocket is None:
                self.rocket = Rocket(num_kernels=self.rocket_n_kernels)
                # Need dummy data to fit (will be replaced with proper fit later)
                dummy_data = np.zeros((1, 3, sample_data.shape[1]))  # Now 3 channels
                self.rocket.fit(dummy_data)
            
            features = self.rocket.transform(rocket_sample)
            return features[0]  # Return first (and only) sample
            
        elif self.method == 'handcrafted':
            features = {}
            for i in range(sample_data.shape[0]):
                times = sample_data[i, :, 0]
                x_coords = sample_data[i, :, 1]
                y_coords = sample_data[i, :, 2]
                
                # Time-based features
                features[f'line{i}_duration'] = times[-1] - times[0]
                features[f'line{i}_time_mean'] = np.mean(times)
                features[f'line{i}_time_std'] = np.std(times)
                
                # Velocity features (dx/dt, dy/dt)
                dt = np.diff(times)
                dx = np.diff(x_coords)
                dy = np.diff(y_coords)
                vx = dx / (dt + 1e-10)  # Add small value to avoid division by zero
                vy = dy / (dt + 1e-10)
                
                features[f'line{i}_vx_mean'] = np.mean(vx)
                features[f'line{i}_vy_mean'] = np.mean(vy)
                features[f'line{i}_vx_std'] = np.std(vx)
                features[f'line{i}_vy_std'] = np.std(vy)
                features[f'line{i}_vx_max'] = np.max(vx)
                features[f'line{i}_vy_max'] = np.max(vy)
                
                # Acceleration features (dv/dt)
                if len(vx) > 1:
                    dvx = np.diff(vx)
                    dvy = np.diff(vy)
                    dt_accel = dt[:-1]
                    ax = dvx / (dt_accel + 1e-10)
                    ay = dvy / (dt_accel + 1e-10)
                    
                    features[f'line{i}_ax_mean'] = np.mean(ax)
                    features[f'line{i}_ay_mean'] = np.mean(ay)
                    features[f'line{i}_ax_std'] = np.std(ax)
                    features[f'line{i}_ay_std'] = np.std(ay)
                
                # Original statistical features
                features[f'line{i}_x_mean'] = np.mean(x_coords)
                features[f'line{i}_y_mean'] = np.mean(y_coords)
                features[f'line{i}_x_std'] = np.std(x_coords)
                features[f'line{i}_y_std'] = np.std(y_coords)
                features[f'line{i}_x_min'] = np.min(x_coords)
                features[f'line{i}_y_min'] = np.min(y_coords)
                features[f'line{i}_x_max'] = np.max(x_coords)
                features[f'line{i}_y_max'] = np.max(y_coords)
                
                # Temporal features
                features[f'line{i}_x_autocorr'] = np.correlate(x_coords, x_coords)[0]
                features[f'line{i}_y_autocorr'] = np.correlate(y_coords, y_coords)[0]
                
                # Spectral features
                fft_x = np.abs(np.fft.fft(x_coords))
                fft_y = np.abs(np.fft.fft(y_coords))
                features[f'line{i}_x_fft_mean'] = np.mean(fft_x[1:len(fft_x)//2])
                features[f'line{i}_y_fft_mean'] = np.mean(fft_y[1:len(fft_y)//2])
            
            return features
            
        elif self.method == 'catch22':
            features = {}
            for i in range(sample_data.shape[0]):
                # Add time features
                times = sample_data[i, :, 0]
                catch22_time = catch22_all(times)
                for feat_name, value in zip(catch22_time['names'], catch22_time['values']):
                    features[f'line{i}_time_{feat_name}'] = value
                
                # Add coordinate features
                for coord, name in [(1, 'x'), (2, 'y')]:
                    data = sample_data[i, :, coord]
                    catch22 = catch22_all(data)
                    for feat_name, value in zip(catch22['names'], catch22['values']):
                        features[f'line{i}_{name}_{feat_name}'] = value
            return features
            
        elif self.method == 'tsfresh':
            # Convert to tsfresh format with time component
            df_list = []
            for line_idx in range(sample_data.shape[0]):
                for point_idx in range(sample_data.shape[1]):
                    df_list.append({
                        'id': line_idx,
                        'time': sample_data[line_idx, point_idx, 0],  # Use actual time
                        'x': sample_data[line_idx, point_idx, 1],
                        'y': sample_data[line_idx, point_idx, 2],
                        'velocity_x': 0,  # Will be calculated below
                        'velocity_y': 0   # Will be calculated below
                    })
            
            # Calculate velocities
            df = pd.DataFrame(df_list)
            df = df.sort_values(['id', 'time'])
            df['velocity_x'] = df.groupby('id')['x'].diff() / df.groupby('id')['time'].diff()
            df['velocity_y'] = df.groupby('id')['y'].diff() / df.groupby('id')['time'].diff()
            df.fillna(0, inplace=True)
            
            features = extract_features(
                df, 
                column_id='id', 
                column_sort='time',
                default_fc_parameters=EfficientFCParameters(),
                n_jobs=1  # Parallelization handled at higher level
            )
            return features
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
    def _extract_features_single(self, sample_data):
        """Extract features for a single sample"""
        if sample_data is None:
            return None
            
        if self.method == 'raw':
            return sample_data  # Return raw time series
            
        elif self.method == 'rocket':
            # Reshape for ROCKET: [1, n_lines*2, n_timesteps]
            n_lines = sample_data.shape[0]
            rocket_sample = np.zeros((1, n_lines*2, sample_data.shape[1]))
            
            for i in range(n_lines):
                rocket_sample[0, 2*i] = sample_data[i, :, 0]  # x coordinates
                rocket_sample[0, 2*i+1] = sample_data[i, :, 1]  # y coordinates
            
            if self.rocket is None:
                self.rocket = Rocket(num_kernels=self.rocket_n_kernels)
                # Need dummy data to fit (will be replaced with proper fit later)
                dummy_data = np.zeros((1, 2, sample_data.shape[1]))
                self.rocket.fit(dummy_data)
            
            features = self.rocket.transform(rocket_sample)
            return features[0]  # Return first (and only) sample
            
        elif self.method == 'handcrafted':
            features = {}
            for i in range(sample_data.shape[0]):
                x_coords = sample_data[i, :, 0]
                y_coords = sample_data[i, :, 1]
                
                # Statistical features
                features[f'line{i}_x_mean'] = np.mean(x_coords)
                features[f'line{i}_y_mean'] = np.mean(y_coords)
                features[f'line{i}_x_std'] = np.std(x_coords)
                features[f'line{i}_y_std'] = np.std(y_coords)
                features[f'line{i}_x_min'] = np.min(x_coords)
                features[f'line{i}_y_min'] = np.min(y_coords)
                features[f'line{i}_x_max'] = np.max(x_coords)
                features[f'line{i}_y_max'] = np.max(y_coords)
                
                # Temporal features
                features[f'line{i}_x_autocorr'] = np.correlate(x_coords, x_coords)[0]
                features[f'line{i}_y_autocorr'] = np.correlate(y_coords, y_coords)[0]
                
                # Spectral features
                fft_x = np.abs(np.fft.fft(x_coords))
                fft_y = np.abs(np.fft.fft(y_coords))
                features[f'line{i}_x_fft_mean'] = np.mean(fft_x[1:len(fft_x)//2])
                features[f'line{i}_y_fft_mean'] = np.mean(fft_y[1:len(fft_y)//2])
            
            return features
            
        elif self.method == 'catch22':
            features = {}
            for i in range(sample_data.shape[0]):
                for coord, name in [(0, 'x'), (1, 'y')]:
                    data = sample_data[i, :, coord]
                    catch22 = catch22_all(data)
                    for feat_name, value in zip(catch22['names'], catch22['values']):
                        features[f'line{i}_{name}_{feat_name}'] = value
            return features
            
        elif self.method == 'tsfresh':
            # Convert to tsfresh format
            df_list = []
            for line_idx in range(sample_data.shape[0]):
                for point_idx in range(sample_data.shape[1]):
                    df_list.append({
                        'id': line_idx,
                        'time': point_idx,
                        'x': sample_data[line_idx, point_idx, 0],
                        'y': sample_data[line_idx, point_idx, 1]
                    })
            
            df = pd.DataFrame(df_list)
            features = extract_features(
                df, 
                column_id='id', 
                column_sort='time',
                default_fc_parameters=EfficientFCParameters(),
                n_jobs=1  # Parallelization handled at higher level
            )
            return features
            
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def process_sample(self, row, save_to_disk=True):
        """Process a single sample from the DataFrame"""
        sample_id = row.uid  # Assuming index is sample ID
        file_path = row['path']  # Assuming column is named 'path'
        feature_path = None  # Initialize feature path
        
        try:
            # Load and preprocess sample
            sample_data = self._load_sample(file_path)
            if sample_data is None:
                return None, None
            
            # Extract features
            features = self._extract_features_single(sample_data)
            
            # Save to disk
            if save_to_disk:
                output_path = os.path.join(self.output_dir, f'{sample_id}.pkl')
                with open(output_path, 'wb') as f:
                    pickle.dump({
                        'sample_id': sample_id,
                        'features': features,
                        'method': self.method,
                        'original_shape': sample_data.shape
                    }, f)
                feature_path = output_path  # Store the path where features were saved
            
            return features, feature_path
        except Exception as e:
            print(f"Error processing sample {sample_id}: {str(e)}")
            return None, None

    def process_dataframe(self, df):
        """Process all samples in the DataFrame"""
        # First pass to fit any necessary transformers
        if self.method == 'rocket':
            print("Fitting ROCKET transformer...")
            dummy_sample = np.zeros((1, 2, 100))  # Assumes minimum 100 timesteps
            self.rocket = Rocket(num_kernels=self.rocket_n_kernels)
            self.rocket.fit(dummy_sample)
        
        # Process samples in parallel
        print(f"Processing {len(df)} samples with {self.method} method...")
        with Pool(processes=self.n_jobs if self.n_jobs != -1 else os.cpu_count()) as pool:
            results = list(pool.imap(
                partial(self.process_sample, save_to_disk=True),
                [row for _, row in df.iterrows()]
            ))
        
        # Extract features and paths from results
        features_list = [r[0] for r in results if r[0] is not None]
        feature_paths = [r[1] for r in results if r[1] is not None]
        
        # Add feature paths to the original dataframe
        df['feature_path'] = None  # Initialize column
        for idx, row in df.iterrows():
            sample_id = row.uid
            for result in results:
                if result[1] and sample_id in result[1]:  # Check if path exists and matches sample_id
                    df.at[idx, 'feature_path'] = result[1]
                    break
        
        return features_list, df  # Return both features and modified dataframe

    