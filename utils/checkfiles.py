# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 22:49:22 2022

@author: mitran
"""
import os

path="./data/scans/"

for i in os.listdir(path):
    for j in os.listdir(path+i):
        if((j.startswith('tmp'))):
            print(i+'/'+j)
    