# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 21:54:12 2022

@author: mitran
"""

import numpy as np
import torch
from os import listdir 
import os
from distutils.dir_util import copy_tree
import shutil


def mkdir(path):
    if(os.path.exists(path)):return
    os.mkdir(path)

path='./scans_val/'
output='./local_val/'
split=27

mkdir(output)
for scene in listdir(path):
    N=len(listdir(path+scene+'/color'))
    print(N)
    
    
    for i in range(N//split):
        name=scene+'_'+str(i)
        mkdir(output+name)
        copy_tree(path+scene+'/'+'intrinsic',output+name+'/intrinsic')
        mkdir(output+name+'/depth')
        mkdir(output+name+'/color')
        mkdir(output+name+'/pose')
        
        for j in range(split):
            
            no=i*split+j
            shutil.copyfile(path+scene+'/color/'+str(no)+'.jpg',output+name+'/color/'+str(j)+'.jpg' )
            shutil.copyfile(path+scene+'/depth/'+str(no)+'.png',output+name+'/depth/'+str(j)+'.png' )
            shutil.copyfile(path+scene+'/pose/'+str(no)+'.txt',output+name+'/pose/'+str(j)+'.txt' )

            
            
            
    
        
        
        
        
        
        
        
        
    
    