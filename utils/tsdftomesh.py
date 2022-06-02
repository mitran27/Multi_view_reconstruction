# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:09:27 2022

@author: mitran
"""

import numpy as np
from skimage import measure
from os import listdir


def meshwrite(filename, verts, faces, norms):
    """Save a 3D mesh to a polygon .ply file.
    """
    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (
            verts[i, 0], verts[i, 1], verts[i, 2],
            norms[i, 0], norms[i, 1], norms[i, 2],
            125, 125, 125,
        ))

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()

"""
source_path='./localtsdfs/local_tsdf_test/'
for scene in listdir(source_path):
    path=source_path+scene+'/'
        
    tsdf=np.load(path+ 'full_tsdf_layer0.npz',allow_pickle=True)['arr_0']
    origin=np.loadtxt(path+ 'voxel_origin.txt', delimiter=' ')
    out_file=path+'mesh.ply'
    
    verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf, level=0)
    verts = verts * 0.04 + origin  
    meshwrite(out_file,verts,faces,norms)
"""

path='D:/scannet/utils/scene0622_01_3/'
        
tsdf=np.load(path+ 'full_tsdf_layer0.npz',allow_pickle=True)['arr_0']
import pickle
origin=np.loadtxt(path+ 'voxel_origin.txt', delimiter=' ')
#with open(path+'tsdf_info.pkl', 'rb') as f:
#            origin = pickle.load(f)['vol_origin']
out_file=path+'mesh.ply'

verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf, level=0)
verts = verts * 0.04 + origin  
meshwrite(out_file,verts,faces,norms)


import open3d as o3d


mesh = o3d.io.read_triangle_mesh(out_file)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([192/255,192/255, 192/255])
o3d.visualization.draw_geometries([mesh])
    
    

