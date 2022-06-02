# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:54:11 2022

@author: mitran
"""

import os
import numpy as np


""" os.remove('./'+mode+'/'+scene+'/color'+'/'+pose.replace('txt','jpg'))
                    os.remove('./'+mode+'/'+scene+'/depth'+'/'+pose.replace('txt','png'))
                    os.remove('./'+mode+'/'+scene+'/pose'+'/'+pose)
                    
                     os.rename(pth+'pose/'+src_name+'.txt', pth+'pose/'+dst_name+'.txt')
                os.rename(pth+'color/'+src_name+'.jpg', pth+'color/'+dst_name+'.jpg')
                os.rename(pth+'depth/'+src_name+'.png', pth+'depth/'+dst_name+'.png')

"""

nanlist=[]

def numsort(x):
    return int(x.split('.')[0])

for mode in os.listdir('./'):
    if(mode.startswith("scans_train")):
        print(mode)
        for scene in os.listdir('./'+mode):
            files= sorted(os.listdir('./'+mode+'/'+scene+'/pose'),key=numsort)
            print(scene)
            assert(len(os.listdir('./'+mode+'/'+scene+'/pose'))==len(os.listdir('./'+mode+'/'+scene+'/depth'))==len(os.listdir('./'+mode+'/'+scene+'/color')))
            """
            pth='./'+mode+'/'+scene+'/'"""
            
            for i,pose in enumerate(files):
               
                
                
                P=np.loadtxt('./'+mode+'/'+scene+'/pose'+'/'+pose, delimiter=' ')
                x=np.isnan(P).any()
                y=np.isinf(P).any()
                if(x or y):
                    nanlist.append('./'+mode+'/'+scene+'/pose'+'/'+pose)
                    
                    
                   
    