import numpy as np
import scipy.io as sio
import os
from scipy.interpolate import Rbf

source_folder = r"C:\Users\15234\Desktop\test"
destination_folder = r"C:\Users\15234\Desktop\test"
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

grid_size = 32
raw_coordinates = {
    'AF3': (-26.051822, 63.01595), 'AF4': (26.051822, 63.01595),
    'F7': (-73.916287, 53.965922), 'F8': (73.962792, 53.908894),
    'F3': (-34.414121, 41.163207), 'F4': (34.460947, 41.146194),
    'FC5': (-61.407155, 23.164645), 'FC6': (61.407155, 23.164645),
    'T7': (-92.741612, 0), 'T8': (92.741612, 0),
    'P7': (-73.916287, -53.965922), 'P8': (73.962792, -53.908894),
    'O1': (-27.222197, -84.150966), 'O2': (27.222197, -84.150966)
}

min_x = min(coord[0] for coord in raw_coordinates.values())
max_x = max(coord[0] for coord in raw_coordinates.values())
min_y = min(coord[1] for coord in raw_coordinates.values())
max_y = max(coord[1] for coord in raw_coordinates.values())

channel_layout = {}
for channel, (x, y) in raw_coordinates.items():
    norm_x = (grid_size - 1) * (x - min_x) / (max_x - min_x)
    norm_y = (grid_size - 1) * (y - min_y) / (max_y - min_y)
    grid_x = int(round(norm_x))
    grid_y = int(round(norm_y))
    channel_layout[channel] = (grid_x, grid_y)

def create_spatial_map(channel_data):
    spatial_map = np.zeros((grid_size, grid_size))
    x_indices = []
    y_indices = []
    values = []
    for channel, position in channel_layout.items():
        x, y = position
        spatial_map[x, y] = channel_data[list(channel_layout.keys()).index(channel)]
        x_indices.append(x)
        y_indices.append(y)
        values.append(channel_data[list(channel_layout.keys()).index(channel)])

    if len(values) > 0:
        rbf = Rbf(x_indices, y_indices, values, function='multiquadric')
        xi, yi = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
        interpolated_values = rbf(xi, yi)
        spatial_map = interpolated_values

    return spatial_map

for i in range(1, 36):
    filename = f"{i}_features.mat"
    filename2 = f"{i}_featuremap.mat"
    mat_contents = sio.loadmat(os.path.join(source_folder, filename))
    data = mat_contents['features']

    time_features = data[:, :224]
    psd_features = data[:, 224:280]
    de_features = data[:, 280:336]

    spatial_maps = np.zeros((360, 24, grid_size, grid_size))

    for sample_index in range(360):
        for feature_index in range(20):
            if feature_index < 16:
                channel_data = time_features[sample_index, feature_index::16]
            else:
                freq_index = feature_index - 16
                channel_data = psd_features[sample_index, freq_index::4]

            spatial_maps[sample_index, feature_index] = create_spatial_map(channel_data)

        for de_index in range(4):
            channel_data = de_features[sample_index, de_index::4]
            spatial_maps[sample_index, 20 + de_index] = create_spatial_map(channel_data)

    sio.savemat(os.path.join(destination_folder, filename2), {'spatial_maps': spatial_maps})

