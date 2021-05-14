import numpy as np
from tqdm import tqdm
import trimesh
import open3d as o3d
import math
import matplotlib.pyplot as plt
import os, re
from omegaconf import OmegaConf
from collections import Counter
from skimage.filters import threshold_minimum
import scipy
import argparse
# %%

# Helper-functions

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text) ]

def fibonacci_sphere(samples=2):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))

    return points

# Nothing to see here
def toIntAppended1D(data):
    int_1d = np.zeros((len(data)))
    for i in range(len(data)):
        s = data[i]
        int_1d[i] = int(str(s[0]) + str(s[1]) + str(s[2]))
    return int_1d

def List2Vec(lst, shape = (320,303,228)):
    vecs = np.reshape(lst, shape)
    vecs = np.array(np.where(vecs!=0)).T.astype(np.int16)
    return vecs

def Vec2List(vecs, shape = (320,303,228)):
    cube = np.zeros(shape)
    cube[vecs[:,0], vecs[:,1], vecs[:,2]] = 1
    lst = np.reshape(cube, -1)
    return lst
    #coords = np.moveaxis(np.indices(shape), 0, -1)
    
def coords2pc(coords, plot=True, color = [1,0,0]):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)

    pcd_colors = np.repeat([color],len(coords), axis=0)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    pcd.estimate_normals()

    # Visualization-check
    if plot:
        o3d.visualization.draw_geometries([pcd])
        
    return pcd

def getNumHitPerCoord(locs, threshold=None):
    # count number of hits for each point in the point-cloud of the heart surface
    c = Counter(map(tuple, locs))

    coords = np.array(list(c.keys()))
    key_values = np.array(list(c.values()))   
    
    s_inds = toIntAppended1D(surface_inds)
    s_coords = toIntAppended1D(coords)
    s, a_inds, b_inds = np.intersect1d(s_inds, s_coords, return_indices=True)
    
    if threshold==None:
        lst = np.concatenate([surface_inds[a_inds], np.array([key_values[b_inds]]).T], axis=1)
    else:
        lst = np.concatenate([surface_inds[a_inds], np.array([key_values[b_inds]]).T>threshold], axis=1)
    return lst  
    
def counter2apearance(coords, values, threshold=12):
    s_inds = toIntAppended1D(surface_inds)
    s_coords = toIntAppended1D(coords)
    s, a_inds, b_inds = np.intersect1d(s_inds, s_coords, return_indices=True)
    
    print(np.array([values[b_inds]]).T.shape, surface_inds[a_inds].shape)
    lst = np.concatenate([surface_inds[a_inds], np.array([values[b_inds]]).T>threshold], axis=1)
    return lst

# %%

if __name__=='__main__': 
    shape = (320,303,228) # set 3D discretisation


    parser = argparse.ArgumentParser(description='Unzipping file and proccessing to smaller numpy-arrays')

    parser.add_argument('-file_surface', '-source', type=str, help='', default='../../data/processed/Heart_surface')
    parser.add_argument('-target_folder', '-source', type=str, help='', default='../../data/processed/')

    args = parser.parse_args() 
    
    config = OmegaConf.create(vars(args))
    print(config)

    surface = np.fromfile(config.file_surface, dtype=np.int32)
    surface_inds = List2Vec(surface) # shape: [m,3], transposed list of 3D-vectors
    
    
    # Generate o3d point cloud from the heart
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_inds)
    colors = np.ones((len(surface_inds), 3))*0.7
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals()
        
    #o3d.visualization.draw_geometries([pcd], point_show_normal=False)  
    
    
    # Generate o3d-mesh from point cloud from the heart
    radii = np.array([0.005, 0.01, 0.02, 0.04])*100
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
   
    
    # Generate Ray-tracer for ray-mesh-intersections
    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), 
                               np.asarray(mesh.triangles),
                               vertex_normals=np.asarray(mesh.vertex_normals))
    
    tracer = trimesh.ray.ray_pyembree.RayMeshIntersector(tri_mesh)
      
    
    # Generate fibonacci points as enveloping sphere around the heart
    num_points = 5000 # number of ray-origins around the heart
    radius = 250 # arbitrary tuned
    s_off = np.array([surface_inds[:,i].mean() for i in range(3)]) + np.array([-10,-20,0]) # manually tuned
    points = np.array(fibonacci_sphere(num_points))*radius + s_off
    
    # Generate o3d-point-cloud from fibonacci-sphere-point-cloud
    sphere = o3d.geometry.PointCloud()
    sphere.points = o3d.utility.Vector3dVector(points)
    sphere.estimate_normals()
     
    # origins are the previously defineds point on the enveloping sphere
    # directions of the rays from ever origin are again points on a sphere:
    num_dirs = 20000 # ( 1 000 000)
    directions = np.array(fibonacci_sphere(num_dirs))
    origins = points   
     
    # find ray-intersections for every origin with every direction
    locs = []
    for origin in tqdm(origins):
        _origins = np.repeat([origin], len(directions), axis=0)
    
        loc, _, _ = tracer.intersects_location(_origins, directions, multiple_hits=False)
        loc = (loc+0.5).astype(np.int16) # round intersection-coordinate to find the closest point at the surface
        locs.append(loc)
    locs = np.concatenate(locs, axis=0)
     
    # counts for each point at the heart surface how often it got hit by rays: [x,y,z,num_hits]
    # if threshold!=None: set num_hits to 1 if number is above threshold
    surface_inds_appearance = getNumHitPerCoord(locs, threshold=None)
    
    s_inds = toIntAppended1D(surface_inds)
    s_coords = toIntAppended1D(surface_inds_appearance[:,:3])
    s, a_inds, b_inds = np.intersect1d(s_inds, s_coords, return_indices=True)
    surface_inds_appearance_real = np.concatenate([surface_inds, np.zeros((len(surface_inds),1))], axis=1).astype(np.int16)
    surface_inds_appearance_real[a_inds,3] = surface_inds_appearance[b_inds,3]
        
    # a bit postprocessing
    cube = np.zeros(shape).astype(np.float32)
    coords = surface_inds_appearance_real[:,:3]
    values = surface_inds_appearance_real[:,-1]
    cube[coords[:,0],coords[:,1],coords[:,2]] = values
    
    # filtering
    cube_filtered = cube.copy()
        
    cube_filtered = scipy.ndimage.convolve(cube_filtered, np.ones((3,3,3))/(3**3))
    cube_filtered = scipy.ndimage.convolve(cube_filtered, np.ones((5,5,5))/(5**3))
    cube_filtered = scipy.ndimage.convolve(cube_filtered, np.ones((7,7,7))/(7**3))
    cube_filtered = scipy.ndimage.convolve(cube_filtered, np.ones((9,9,9))/(9**3))
    cube_filtered = scipy.ndimage.convolve(cube_filtered, np.ones((11,11,11))/(11**3))
    
    new_values = cube_filtered[coords[:,0],coords[:,1],coords[:,2]]
    plt.hist(new_values, bins=64)
    plt.show()
    
    thresh_min = threshold_minimum(new_values)
    threshold = thresh_min
            
    new_values = cube_filtered[coords[:,0],coords[:,1],coords[:,2]]
    plt.hist(new_values, bins=64)
    plt.show()
    
    # %% 
    
    outer = coords2pc(coords[new_values>1.4], plot=0, color=[1,0,0]) # threshold manually tuned
    inner = coords2pc(coords[new_values<=1.4], plot=0, color=[0,0,1])
    o3d.visualization.draw_geometries([outer, inner])
    
    # %%
    outer_surface_list = Vec2List(coords[new_values>1.4]).astype(np.int8)
    inner_surface_list = Vec2List(coords[new_values<=1.4]).astype(np.int8)
    
    target_file_outer = os.path.join(config.target_folder, 'outer_surface')
    target_file_inner = os.path.join(config.target_folder, 'inner_surface')
    
    np.save(target_file_outer, outer_surface_list)
    np.save(target_file_inner, inner_surface_list)