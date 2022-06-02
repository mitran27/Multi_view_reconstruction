# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 07:16:05 2022

@author: mitran
"""

import torch

feats=torch.randn(1,4,10,10)


n_vox=[10,5]
grid_range = [torch.arange(0, n_vox[axis], 1) for axis in range(2)]
grid = torch.stack(torch.meshgrid(grid_range[0], grid_range[1],indexing='ij'),dim=-1).float()

"""
x=grid[2][4]
x[0]=(x[0]/9 -0.5)*2
x[1]=(x[1]/9 -0.5)*2


x=x.view(1,1,1,2)
y=torch.nn.functional.grid_sample(feats, x, padding_mode='zeros',align_corners=True)
print(y)
print(feats[0,:,2,4])"""
