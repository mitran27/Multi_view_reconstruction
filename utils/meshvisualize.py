# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 18:10:05 2022

@author: mitran
"""
import open3d as o3d
from os import listdir
import random
#path='./localtsdfs/local_tsdf_test/'
#file=path+random.choice(listdir(path))+'/mesh.ply'
#print(file)
#path="./scene.ply"
file='./pred4.ply'
display= 0

if(display==1):

    pcd = o3d.io.read_point_cloud(file)
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
else:
    mesh = o3d.io.read_triangle_mesh(file)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([192/255,192/255, 192/255])
    o3d.visualization.draw_geometries([mesh])
    
    

    