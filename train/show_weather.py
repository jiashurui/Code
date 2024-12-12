import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file to check its structure
data_path = "../data/weather/Data.csv"
data = pd.read_csv(data_path)

# Extracting data for January 1, 2022
date_data = data[data['日付'] == '2022-10-29']

# Dropping non-relevant columns for the visualization
pressure_data = date_data.drop(columns=['日付', '沖縄の天気', '沖縄の降水量'])

# Extract coordinates and their corresponding pressure values
coordinates = [tuple(map(float, col.split('_'))) for col in pressure_data.columns]
latitudes, longitudes = zip(*coordinates)
pressure_values = pressure_data.values.flatten()

# Create a grid for plotting
unique_latitudes = np.unique(latitudes)
unique_longitudes = np.unique(longitudes)
latitude_indices = np.searchsorted(unique_latitudes, latitudes)
longitude_indices = np.searchsorted(unique_longitudes, longitudes)

# Creating a mesh grid for latitude and longitude
grid = np.zeros((len(unique_latitudes), len(unique_longitudes)))
grid[latitude_indices, longitude_indices] = pressure_values

# Filtering data for specified latitude and longitude ranges
latitude_min, latitude_max = 30, 46
longitude_min, longitude_max = 128, 144

# Indices for the desired latitude and longitude ranges
lat_range_indices = (unique_latitudes >= latitude_min) & (unique_latitudes <= latitude_max)
lon_range_indices = (unique_longitudes >= longitude_min) & (unique_longitudes <= longitude_max)

# Filtering the latitude and longitude arrays
filtered_latitudes = unique_latitudes[lat_range_indices]
filtered_longitudes = unique_longitudes[lon_range_indices]

# Creating a new grid for the filtered data
filtered_grid = grid[np.ix_(lat_range_indices, lon_range_indices)]

# Adjusting the colormap to a grayscale to represent the data in black and white
plt.figure(figsize=(10, 7))
plt.imshow(filtered_grid, cmap='gray',
           extent=[filtered_longitudes.min(), filtered_longitudes.max(), filtered_latitudes.min(), filtered_latitudes.max()], origin='lower')
plt.colorbar(label='Air Pressure (Pa)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()