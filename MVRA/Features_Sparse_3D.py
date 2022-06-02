from torch.nn import Module,Sequential
import torch
import torchsparse.nn as spnn
import torchsparse


class Sparse_3d_Convolution_Block(Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride=1):
        super().__init__()
        self.model = Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,dilation=1),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.model(x)

class Sparse_3d_Residual_Block(Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride=1):
        super().__init__()
        self.model = Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,dilation=1),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
            spnn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride,dilation=1),
            spnn.BatchNorm(out_channels),
        )
        self.Residual =Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride,dilation=1),
            spnn.BatchNorm(out_channels),
        )
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        return self.relu(self.model(x) + self.Residual(x))

class Unet_Up(Module):
  

  def __init__(self,**kwargs):
    super().__init__()
    in_dim = kwargs['input_dim']
    out_dim = kwargs['output_dim']
    inter_dim = kwargs['intermediate_dim']
    self.TransposedConv = Sequential(
            spnn.Conv3d(in_dim, in_dim, kernel_size=2, stride=2, transposed=True),
            spnn.BatchNorm(in_dim),
            spnn.ReLU(True),
        )
    self.Conv = Sequential(
          Sparse_3d_Residual_Block(in_dim + inter_dim , out_dim,kernel_size=3,stride=1),
          Sparse_3d_Residual_Block(out_dim,out_dim,kernel_size=3,stride=1)
    )
  def forward(self,Input,intermediate):

    y = self.TransposedConv(Input)
    y = torchsparse.cat([y, intermediate])
    y = self.Conv(y)

    return y


class Unet_3d_Sparse(Module):
  # spatial refinement
  # sparse COnvolutions
  # residual Unet

  def __init__(self,Features_channels,Block_dim_list,residual=False):
    super().__init__()


    self.Base = Sparse_3d_Convolution_Block(Features_channels,Block_dim_list[0],3)
    
    
    # Unet Down : 1) Downsampling 2)Channel dimension change 3)  refinement
    self.Down1 = Sequential(
          Sparse_3d_Convolution_Block(Block_dim_list[0],Block_dim_list[0],kernel_size=2,stride=2),
          Sparse_3d_Residual_Block(Block_dim_list[0],Block_dim_list[1],kernel_size=3,stride=1),
          Sparse_3d_Residual_Block(Block_dim_list[1],Block_dim_list[1],kernel_size=3,stride=1)
    )
    self.Down2 = Sequential(
          Sparse_3d_Convolution_Block(Block_dim_list[1],Block_dim_list[1],kernel_size=2,stride=2),
          Sparse_3d_Residual_Block(Block_dim_list[1],Block_dim_list[2],kernel_size=3,stride=1),
          Sparse_3d_Residual_Block(Block_dim_list[2],Block_dim_list[2],kernel_size=3,stride=1)
    )

    # Unet Up = 1) Upsampling 2) Concat 3)Channel dimension change 4) refinement 
    self.Up1 = Unet_Up(input_dim=Block_dim_list[2],output_dim=Block_dim_list[3],intermediate_dim=Block_dim_list[1])
    self.Up2 = Unet_Up(input_dim=Block_dim_list[3],output_dim=Block_dim_list[4],intermediate_dim=Block_dim_list[0])
    
    self.dropout = torch.nn.Dropout(0.5)

  def forward(self,voxel_volume):
    y = self.Base(voxel_volume)
    
    y1 = self.Down1(y)
    y2 = self.Down2(y1)
    
    y2.F = self.dropout(y2.F)

    yu1 = self.Up1(y2 , intermediate = y1)
    yu2 = self.Up2(yu1 , intermediate = y)
    
    return yu2









