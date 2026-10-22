import numpy as np
from PIL import Image as img
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import plotly.express as px
from skimage.transform import resize
from plyfile import PlyData
from sklearn.neighbors import KDTree
import pandas as pd
from math import sqrt
from math import pi
from math import acos
import os

_PROJECTION_RADIUS = 0.1 #0.0425

# label value is semantic_ID * 1000 + instance_ID (no need to change for KITTI)

_semantic_ID_dict = {
    0 : "unlabeled",
    1 : "ego vehicle",
    2 : "rectification border",
    3 : "out of roi",
    4 : "static",
    5 : "dynamic",
    6 : "ground",
    7 : "road",
    8 : "sidewalk",
    9 : "parking",
    10 : "rail track",
    11 : "building",
    12 : "wall",
    13 : "fence",
    14 : "guard rail",
    15 : "bridge",
    16 : "tunnel",
    17 : "pole",
    18 : "polegroup",
    19 : "traffic light",
    20 : "traffic sign",
    21 : "vegetation",
    22 : "terrain",
    23 : "sky",
    24 : "person",
    25 : "rider",
    26 : "car",
    27 : "truck",
    28 : "bus",
    29 : "caravan",
    30 : "trailer",
    31 : "train",
    32 : "motorcycle",
    33 : "bicycle",
    34 : "garage",
    35 : "gate",
    36 : "stop",
    37 : "smallpole",
    38 : "lamp",
    39 : "trash bin",
    40 : "vending machine",
    41 : "box",
    42 : "unknown construction",
    43 : "unknown vehicle",
    44 : "unknown object",
    45 : "license plate"
}

def _get_KITTI_labeled_image_files(agnostic_image_directory):
    image_files = list(Path(agnostic_image_directory).glob('*.png'))
    return image_files

def _get_KITTI_labeled_mesh_files(agnostic_mesh_directory):
    mesh_files = list(Path(agnostic_mesh_directory).glob('*.ply'))
    return mesh_files

def _get_KITTI_raw_scan_files(agnostic_scan_directory):
    scan_files = list(Path(agnostic_scan_directory).glob('*.bin'))
    return scan_files


def _get_KITTI_point_cloud_xyz(scan_file):
    return np.reshape(np.fromfile(scan_file, dtype = np.float32), (-1, 4)) # last dimension is, x, y, z, reflectivity


def _xyz_to_cylindrical(point_cloud_xyz : np.array, LIDAR_relative_xyz : tuple) -> np.array:
    '''
    takes array of shape (point_num, 4) with the last dimension being x, y, z, reflectivity and tuple of shape (3) for the x y z adjustments
    returns (index, z, theta, range, reflectivity)
    '''

    point_cloud_cylindrical = np.full(fill_value= -1.0, shape = (point_cloud_xyz.shape[0], 5))

    projection_x  = LIDAR_relative_xyz[0]
    projection_y = LIDAR_relative_xyz[1]
    projection_z = LIDAR_relative_xyz[2]

    for i in range(point_cloud_xyz.shape[0]):
        point_cloud_cylindrical[i,0] = float(i)
        cylinder_z = 0.0
        theta = 0.0        
        point_range = 0.0
        reflectivity = point_cloud_xyz[i,3]

        original_x = point_cloud_xyz[i,0]
        original_y = point_cloud_xyz[i,1]
        original_z = point_cloud_xyz[i,2]

        if (sqrt((original_x-projection_x)**2 +(original_y - projection_y)**2) <= _PROJECTION_RADIUS):
            #point_cloud_cylindrical[i,1] = point_cloud_cylindrical[i,2] = point_cloud_cylindrical[i,3] = point_cloud_cylindrical[i,4] = -1.0
            continue

        cylinder_x = original_x + (projection_x- original_x)*(1-sqrt(_PROJECTION_RADIUS**2/((original_x-projection_x)**2+(original_y-projection_y)**2)))
        cylinder_y = original_y + (projection_y - original_y)*(1-sqrt(_PROJECTION_RADIUS**2/((original_x-projection_x)**2+(original_y-projection_y)**2)))
        cylinder_z = original_z + (projection_z - original_z)*(1-sqrt(_PROJECTION_RADIUS**2/((original_x-projection_x)**2+(original_y-projection_y)**2)))

        cosine = (cylinder_x-projection_x)/_PROJECTION_RADIUS
        sine = (cylinder_y-projection_y)/_PROJECTION_RADIUS

        if cosine >= 1.0:
            cosine -= 0.001
        elif cosine <= -1.0:
            cosine += 0.001

        if sine >= 1.0:
            sine -= 0.001
        elif sine <= -1.0:
            sine += 0.001
        if sine > 0:
            theta = acos(cosine)
        else:
            theta = 2*pi - acos(cosine)

        theta += pi
        if theta > 2*pi:
            theta -= 2*pi

        point_range = sqrt((original_x-projection_x)**2+(original_y-projection_y)**2+(original_z-projection_z)**2)

        point_cloud_cylindrical[i,1] = cylinder_z
        point_cloud_cylindrical[i,2] = theta
        point_cloud_cylindrical[i,3] = point_range
        point_cloud_cylindrical[i,4] = reflectivity

    
    point_cloud_cylindrical = point_cloud_cylindrical[point_cloud_cylindrical[...,0] > -1.0]
    return point_cloud_cylindrical



def _cylindrical_to_labeled_point_cloud_xyz(point_cloud_cylindrical, point_cloud_xyz, labeled_image, image_margins : tuple, LIDAR_margins : tuple, label_decode_function, instance_decode_function) -> np.array:  # DOESN'T ACCOUNT FOR IMAGE THAT EXTENDS BEYOND LIDAR'S 0 DEGREES
    '''
    input array needs to be of shape (point_num, 5) with the last one being index, z, theta, range, reflectivity
    '''
    
    label_matrix = label_decode_function(labeled_image)
    instance_matrix = instance_decode_function(labeled_image)

    labeled_point_cloud_xyz = np.full(fill_value= -1.0, shape = (point_cloud_xyz.shape[0], point_cloud_xyz.shape[1]+2))
    labeled_point_cloud_xyz[...,:-2] = point_cloud_xyz

    assignment_cutoff = np.max(point_cloud_cylindrical[...,-2]) +1 # ranges (doesn't change)

    projection_size_y = int(np.ptp(point_cloud_cylindrical[...,1])*17379.3*(0.0425/_PROJECTION_RADIUS)) # arbitrary values
    projection_size_x = 2048

    projection = np.full(fill_value= assignment_cutoff, shape = (projection_size_y, projection_size_x))
    y_edges = np.linspace(np.min(point_cloud_cylindrical[...,1]), np.max(point_cloud_cylindrical[...,1])+np.ptp(point_cloud_cylindrical[...,1])/projection_size_y, projection_size_y)
    x_edges = np.linspace(np.min(point_cloud_cylindrical[...,2]), np.max(point_cloud_cylindrical[...,2])+np.ptp(point_cloud_cylindrical[...,2])/projection_size_x, projection_size_x)
    y_indices = np.digitize(point_cloud_cylindrical[...,1], y_edges) -1
    x_indices = np.digitize(point_cloud_cylindrical[...,2], x_edges) -1

    projection_raveled = projection.ravel()
    indices_raveled = np.ravel_multi_index((y_indices,x_indices), projection.shape)
    for i in range(np.ravel_multi_index((y_indices,x_indices), projection.shape).shape[0]):
        if projection_raveled[indices_raveled[i]] > point_cloud_cylindrical[i,-2]: # ranges (doesn't change)
            projection_raveled[indices_raveled[i]] = point_cloud_cylindrical[i,0]

    projection[projection == assignment_cutoff] = -1.0
    projection = np.flip(projection, axis = 0) # Because z and image height are flipped (higher the z, lower the axis 0 value)
    projection = projection[int((projection.shape[0] - 1.0)*(LIDAR_margins[0])):int((projection.shape[0] - 1.0)*(1.0-LIDAR_margins[1]))]
    projection = projection[...,int((projection.shape[1] - 1.0)*(LIDAR_margins[2]/360.0)):int((projection.shape[1] - 1.0)*(LIDAR_margins[3]/360.0))]
    projection = np.flip(projection, axis = 1) # because angle is counterclockwise

    #aligned_labeled_image = labeled_image[int((labeled_image.shape[0]-2.0)*image_margins[0]): int((labeled_image.shape[0]-2.0)*(1.0-image_margins[1]))]
    #aligned_labeled_image = labeled_image[...,int((labeled_image.shape[1]-2.0)*image_margins[2]):int((labeled_image.shape[1]-2.0)*(1.0-image_margins[3]))]

    label_matrix = label_matrix[int((labeled_image.shape[0]-2.0)*image_margins[0]): int((labeled_image.shape[0]-2.0)*(1.0-image_margins[1]))]
    label_matrix = label_matrix[...,int((labeled_image.shape[1]-2.0)*image_margins[2]):int((labeled_image.shape[1]-2.0)*(1.0-image_margins[3]))]
    label_matrix = resize(label_matrix, output_shape = (projection.shape[0], projection.shape[1]), anti_aliasing=False, preserve_range=True, order = 0)

    instance_matrix = instance_matrix[int((labeled_image.shape[0]-2.0)*image_margins[0]): int((labeled_image.shape[0]-2.0)*(1.0-image_margins[1]))]
    instance_matrix = instance_matrix[...,int((labeled_image.shape[1]-2.0)*image_margins[2]):int((labeled_image.shape[1]-2.0)*(1.0-image_margins[3]))]
    instance_matrix = resize(instance_matrix, output_shape = (projection.shape[0], projection.shape[1]), anti_aliasing=False, preserve_range=True, order = 0)

    #aligned_labeled_image = resize(aligned_labeled_image, output_shape = (projection.shape[0], projection.shape[1]), anti_aliasing=False, preserve_range=True, order = 0)

    projection_raveled = projection.ravel()
    label_matrix_raveled = label_matrix.ravel()
    instance_matrix_raveled = instance_matrix.ravel()
    #aligned_labeled_image_raveled = aligned_labeled_image.ravel()
    #labeled_point_cloud_xyz[projection_raveled[projection_raveled != -1].astype(int),-1] = aligned_labeled_image_raveled[projection_raveled != -1]
    labeled_point_cloud_xyz[projection_raveled[projection_raveled != -1].astype(int),-2] = label_matrix_raveled[projection_raveled != -1]
    labeled_point_cloud_xyz[projection_raveled[projection_raveled != -1].astype(int),-1] = instance_matrix_raveled[projection_raveled != -1]
    
    return labeled_point_cloud_xyz



def _get_decoded_KITTI_label(raw_labeled_image : np.array) -> np.array:
    return (raw_labeled_image/1000.0).astype(int)



def _get_decoded_KITTI_instance(raw_labeled_image : np.array) -> np.array:
    return raw_labeled_image - (raw_labeled_image/1000).astype(int)*1000



def _get_sampled_image(sample_labeled_image, image_margins : tuple,) -> np.array:
    sample_labeled_image = sample_labeled_image[int((sample_labeled_image.shape[0]-2.0)*image_margins[0]): int((sample_labeled_image.shape[0]-2.0)*(1.0-image_margins[1]))]
    sample_labeled_image =  sample_labeled_image[...,int((sample_labeled_image.shape[1]-2.0)*image_margins[2]):int((sample_labeled_image.shape[1]-2.0)*(1.0-image_margins[3]))]

    return sample_labeled_image



def _get_sampled_point_cloud(sample_point_cloud_cylindrical, LIDAR_margins : tuple, return_element : str = 'range') -> np.array: # DOESN'T ACCOUNT FOR IMAGE THAT EXTENDS BEYOND 0 DEGREES
    '''
    input array needs to be of shape (point_num, 5) with the last one being index, z, theta, range, reflectivity
    '''

    return_element_dict = {
        'range' : -2,
        'reflectivity' : -1,
        'index' : 0
    }

    assignment_cutoff = np.max(sample_point_cloud_cylindrical[...,-2]) +1 # ranges (doesn't change)

    projection_size_y = int(np.ptp(sample_point_cloud_cylindrical[...,1])*17379.3*(0.0425/_PROJECTION_RADIUS)) # arbitrary values
    projection_size_x = 2048
    sample_point_cloud_projected = np.full(fill_value= assignment_cutoff, shape = (projection_size_y, projection_size_x))
    y_edges = np.linspace(np.min(sample_point_cloud_cylindrical[...,1]), np.max(sample_point_cloud_cylindrical[...,1])+np.ptp(sample_point_cloud_cylindrical[...,1])/projection_size_y, projection_size_y)
    x_edges = np.linspace(np.min(sample_point_cloud_cylindrical[...,2]), np.max(sample_point_cloud_cylindrical[...,2])+np.ptp(sample_point_cloud_cylindrical[...,2])/projection_size_x, projection_size_x)
    y_indices = np.digitize(sample_point_cloud_cylindrical[...,1], y_edges) -1
    x_indices = np.digitize(sample_point_cloud_cylindrical[...,2], x_edges) -1

    

    sample_point_cloud_projected_raveled = sample_point_cloud_projected.ravel()
    indices_raveled = np.ravel_multi_index((y_indices,x_indices), sample_point_cloud_projected.shape)
    for i in range(np.ravel_multi_index((y_indices,x_indices), sample_point_cloud_projected.shape).shape[0]):
        if sample_point_cloud_projected_raveled[indices_raveled[i]] > sample_point_cloud_cylindrical[i,-2]: # ranges (doesn't change)
            sample_point_cloud_projected_raveled[indices_raveled[i]] = sample_point_cloud_cylindrical[i,return_element_dict[return_element]] # changes depending on desired display
    

    sample_point_cloud_projected[sample_point_cloud_projected == assignment_cutoff] = -1.0

    sample_point_cloud_projected = np.flip(sample_point_cloud_projected, axis = 0) # Because z and image height are flipped (higher the z, lower the axis 0 value)

    sample_point_cloud_projected = sample_point_cloud_projected[int((sample_point_cloud_projected.shape[0] - 1.0)*(LIDAR_margins[0])):int((sample_point_cloud_projected.shape[0] - 1.0)*(1.0-LIDAR_margins[1]))]
    sample_point_cloud_projected = sample_point_cloud_projected[...,int((sample_point_cloud_projected.shape[1] - 1.0)*(LIDAR_margins[2]/360.0)):int((sample_point_cloud_projected.shape[1] - 1.0)*(LIDAR_margins[3]/360.0))]

    sample_point_cloud_projected = np.flip(sample_point_cloud_projected, axis = 1) # because angle is counterclockwise

    return sample_point_cloud_projected



def project_2d_semantic_instance_data(absolute_image_directory : str, absolute_scan_directory: str, absolute_destination_directory : str, prefix : str, LIDAR_relative_xyz : tuple , image_margins : tuple, LIDAR_margins : tuple, data_format : str, sample_only : bool = False, ignore_duplicates : bool = False) -> int:
    ''' 
    WARNING: MAKE SURE TO HAVE CLEANED YOUR IMAGE AND SCAN DATA BEFORE RUNNING THIS FUNCTION! MULTIPLE RUNS WILL CAUSE DUPLICATE DATA GENERATION!
    LIDAR_relative_xyz is (x,y,z), camera's position relative to the lidar. X is lidar forward at 0 degrees, Y is left, Z is up in meters
    camera margins is (up, down, left, right), part of screen left out
    LIDAR_margins is (up, down, start_angle, end_angle). up and down are part of screen left out, start_angle and end_angle are the angles in degrees'''
    # Need to do validation for margin inputs
    # label value is semantic_ID * 1000 + instance_ID

    get_image_files = {
        'KITTI' : _get_KITTI_labeled_image_files,
    }

    get_scan_files = {
        'KITTI' : _get_KITTI_raw_scan_files,
    }

    get_point_cloud_xyz = {
        'KITTI' : _get_KITTI_point_cloud_xyz,
    }
    
    agnostic_destination_directory = os.path.normpath(absolute_destination_directory)

    existing_save_files = list(Path(agnostic_destination_directory).glob('*.npy'))

    agnostic_image_directory = os.path.normpath(absolute_image_directory)
    agnostic_scan_directory = os.path.normpath(absolute_scan_directory)

    image_files = get_image_files[data_format](agnostic_image_directory)

    scan_files = get_scan_files[data_format](agnostic_scan_directory)

    matching_files = [] # Will add tuples that go (image, scan)

    for i in range(len(image_files)):
        for n in range(len(scan_files)):
            if image_files[i].name[:-3] == scan_files[n].name[:-3]:
                matching_files.append((image_files[i], scan_files[n]))
                if(sample_only == True):
                    n = len(scan_files) -1 
                    i = len(image_files) - 1
                break

    if len(matching_files) == 0:
        return 1 # empty  
 
    for i in range(len(matching_files)):
        print(f"\rCollecting 2D file {i+1}/{len(matching_files)}", end= "")

        continue_flag = False
        save_name = matching_files[i][0].name[:-4]
        for item in existing_save_files:
            if item.name[:-4] == save_name:
                if (ignore_duplicates != True):
                    save_name += "d" # d for duplicate
                    break
                else:
                    continue_flag = True
                    break
        if continue_flag:
            continue
        
        labeled_image = 0
        with img.open(matching_files[i][0]) as file:
            labeled_image = np.array(file)

        point_cloud_xyz = get_point_cloud_xyz[data_format](matching_files[i][1].resolve())
        point_cloud_cylindrical = _xyz_to_cylindrical(point_cloud_xyz, LIDAR_relative_xyz)

        if(sample_only == True):
            sample_labeled_image = _get_sampled_image(labeled_image, image_margins)
            sample_point_cloud_projected = _get_sampled_point_cloud(point_cloud_cylindrical, LIDAR_margins)

            sample_labeled_image_resized = resize(sample_labeled_image, output_shape = (sample_point_cloud_projected.shape[0], sample_point_cloud_projected.shape[1]), anti_aliasing=False, preserve_range=True, order = 0)
            fig, ax = plt.subplots()
            ax.axis('off')
            im1 = ax.imshow(sample_labeled_image_resized, cmap = 'magma', aspect = '1')
            im2 = ax.imshow(sample_point_cloud_projected, cmap = 'magma', aspect = '1')
            im2.set_visible(False)
            current_state = 0
            def toggle(event):
                nonlocal current_state
                if current_state == 0:
                    im1.set_visible(False)
                    im2.set_visible(True)
                    ax.set_title('LIDAR')
                    current_state = 1
                else:
                    im2.set_visible(False)
                    im1.set_visible(True)
                    ax.set_title('Camera')
                    current_state = 0
                fig.canvas.draw()
            ax_button = plt.axes([0.81, 0.05, 0.1, 0.075])
            button = Button(ax_button, 'Switch')
            button.on_clicked(toggle)
            ax.set_title('Camera')

            plt.show()

            return 0
        
        labeled_point_cloud_xyz = _cylindrical_to_labeled_point_cloud_xyz(point_cloud_cylindrical, point_cloud_xyz, labeled_image, image_margins, LIDAR_margins, _get_decoded_KITTI_label, _get_decoded_KITTI_instance)

        save_name = matching_files[i][0].name[:-4]
        
        if (ignore_duplicates == False):
            while(True):
                for item in existing_save_files:
                    if item.name[:-4] == save_name:
                        save_name += "d" # d for duplicate
                        break
                
        os.makedirs(agnostic_destination_directory, exist_ok=True)
        np.save(os.path.join(agnostic_destination_directory, (prefix+save_name)),labeled_point_cloud_xyz)

    print("")
    
    return 0



def inspect_collected_2d_semantic_instance_data(absolute_destination_directory : str, file : int = 0, display_as_3d : bool = False) -> int:
    agnostic_destination_directory = os.path.normpath(absolute_destination_directory)
    existing_save_files = list(Path(agnostic_destination_directory).glob('*.npy'))

    if len(existing_save_files) == 0:
        return 1 # empty

    labeled_point_cloud_xyz = np.load(existing_save_files[file])

    unlabeled_point_cloud_cylindrical = _xyz_to_cylindrical(labeled_point_cloud_xyz[...,:-1], (0.0, 0.0, 0.0))

    projection = _get_sampled_point_cloud(unlabeled_point_cloud_cylindrical, (0.0, 0.0, 0.0, 360.0), return_element = 'index')

    
    projection =  np.reshape(labeled_point_cloud_xyz[projection.ravel().astype(int), -1], shape = projection.shape)

    plt.figure(1)
    plt.imshow(projection, cmap = 'magma', aspect = '1')
    plt.axis('off')
    plt.title('Labeled LIDAR')
    plt.show()

    if display_as_3d:
        fig = px.scatter_3d(
            x = labeled_point_cloud_xyz[...,0],
            y = labeled_point_cloud_xyz[...,1],
            z = labeled_point_cloud_xyz[...,2],
            color = labeled_point_cloud_xyz[...,-1],
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            scene=dict(
                aspectmode='data'
            )
        )
        fig.show()

    return 0



def project_KITTI_3d_semantic_instance_data(absolute_mesh_directory : str, absolute_scan_directory : str, absolute_destination_directory : str, prefix : str, absolute_calibration_directory : str, absolute_poses_directory : str, ignore_duplicates : bool = False) -> int:
    agnostic_cam_to_velo_path = os.path.join(os.path.normpath(absolute_calibration_directory), "calib_cam_to_velo.txt")
    agnostic_cam_to_pose_path = os.path.join(os.path.normpath(absolute_calibration_directory), "calib_cam_to_pose.txt")
    agnostic_poses_path = os.path.join(os.path.normpath(absolute_poses_directory), "poses.txt")
    agnostic_destination_directory = os.path.normpath(absolute_destination_directory)
    agnostic_static_mesh_directory = os.path.join(os.path.normpath(absolute_mesh_directory), "static")
    agnostic_scan_directory = os.path.normpath(absolute_scan_directory)

    static_mesh_files = _get_KITTI_labeled_mesh_files(agnostic_static_mesh_directory)
    scan_files = _get_KITTI_raw_scan_files(agnostic_scan_directory)

    matching_files = [] # Will add tuples that go (static_mesh, scan)
    lengths = []

    existing_save_files = list(Path(agnostic_destination_directory).glob('*.npy'))
       
    for i in range(len(static_mesh_files)):
        lengths.append(0)
        for n in range(len(scan_files)):
            if i > 0:
                for m in range(len(matching_files)-lengths[i]-lengths[i-1], len(matching_files)-lengths[i]):
                    if scan_files[n] == matching_files[m][1]:
                        continue
            try:
                if int(scan_files[n].name[:-4]) >= int(static_mesh_files[i].name[:10]) and int(scan_files[n].name[:-4]) <= int(static_mesh_files[i].name[11:-4]):
                    matching_files.append((static_mesh_files[i], scan_files[n]))
                    lengths[i] += 1
            except ValueError:
                continue

    T_velo_to_cam = np.linalg.inv(  np.vstack([np.loadtxt(agnostic_cam_to_velo_path).reshape(3,4), [0,0,0,1]])  )[:3, :]
    T_cam_to_pose = np.loadtxt(agnostic_cam_to_pose_path, usecols = range(1, 13), max_rows=1).reshape(3,4)
    T_pose_to_world_indices = np.loadtxt(agnostic_poses_path, usecols = range(0,1)).reshape(-1)
    T_pose_to_world_list = np.loadtxt(agnostic_poses_path, usecols = range(1, 13)).reshape(-1,3,4)
    current_static_mesh_file = 0
    static_mesh = 0
    n=0
    len_static_mesh_files = len(static_mesh_files)
    for i in range(len(matching_files)):
        if matching_files[i][0] != current_static_mesh_file:
            n += 1
            print(f"\rprojecting 3D segment {n}/{len_static_mesh_files}", end= "")
            current_static_mesh_file = matching_files[i][0]
            ply = PlyData.read(os.path.join(agnostic_static_mesh_directory, matching_files[i][0].name))
            vertex = ply['vertex']
            x = vertex['x']
            y = vertex['y']
            z = vertex['z']
            semantic = vertex['semantic']
            instance = vertex['instance']
            static_mesh = np.vstack([x, y, z, semantic, instance]).T

        if len(static_mesh) == 0:
            continue

        point_cloud_xyz = _get_KITTI_point_cloud_xyz(matching_files[i][1])
        point_cloud_xyz_transformed = point_cloud_xyz.copy()[...,:3]
        labeled_point_cloud_xyz = np.full(fill_value=-1.0, shape = (point_cloud_xyz.shape[0], point_cloud_xyz.shape[1]+2))
        labeled_point_cloud_xyz[...,:-2] = point_cloud_xyz

        T_pose_to_world = T_pose_to_world_list[T_pose_to_world_indices.astype(int) == int(matching_files[i][1].name[:-4])]
        
        if len(T_pose_to_world) != 1:
            continue
        T_pose_to_world = T_pose_to_world.reshape(3,4)

        T_velo_to_world =  np.vstack([T_pose_to_world, [0,0,0,1]]) @ np.vstack([T_cam_to_pose, [0,0,0,1]]) @ np.vstack([T_velo_to_cam, [0,0,0,1]])

        #point_cloud_xyz_transformed = (T_velo_to_world @ point_cloud_xyz_transformed.T).T[:, :3]
        points_hom = np.hstack([point_cloud_xyz_transformed, np.ones((point_cloud_xyz_transformed.shape[0], 1))])
        transformed_hom = (T_velo_to_world @ points_hom.T).T
        point_cloud_xyz_transformed = transformed_hom[:, :3]
        #point_cloud_xyz_transformed = np.column_stack([transformed_hom[:, :3], np.full(fill_value=-1.0, shape = (point_cloud_xyz_transformed.shape[0],))])

        '''fig = px.scatter_3d(point_cloud_xyz_transformed, 
                            x=0, 
                            y=1, 
                            z=2,
                            color=3, 
                            title='LIDAR')
        fig.update_layout(scene=dict(aspectmode='data'))
        fig.show()'''



        '''fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(static_mesh[:, 0], static_mesh[:, 1], static_mesh[:, 2],
                c=static_mesh[:, 3], cmap='viridis', s=1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("PLY Point Cloud")
        ax.set_aspect('equal')
        plt.show()'''

        tree = KDTree(static_mesh[...,:-2], leaf_size=32, metric='euclidean')
        dist, idx = tree.query(point_cloud_xyz_transformed, k=1)
        labeled_point_cloud_xyz[...,4] = static_mesh[idx[:,0],-2]
        labeled_point_cloud_xyz[...,5] = static_mesh[idx[:,0],-1]-static_mesh[idx[:,0],-2]*1000

        '''fig = px.scatter_3d(labeled_point_cloud_xyz, 
                            x=0, 
                            y=1, 
                            z=2,
                            color=5, 
                            title='LIDAR')
        fig.update_layout(scene=dict(aspectmode='data'))
        fig.show()'''
        
        os.makedirs(agnostic_destination_directory, exist_ok=True)
        np.save(os.path.join(agnostic_destination_directory, f"{matching_files[i][1].name[:-3]}npy"), labeled_point_cloud_xyz)

        save_name = matching_files[i][1].name[:-4]

        if (ignore_duplicates == False):
            while(True):
                for item in existing_save_files:
                    if item.name[:-4] == save_name:
                        save_name += "d" # d for duplicate
                        break
            
        os.makedirs(agnostic_destination_directory, exist_ok=True)
        np.save(os.path.join(agnostic_destination_directory, (prefix+save_name)),labeled_point_cloud_xyz)

    print("")

    return 0



        






        

