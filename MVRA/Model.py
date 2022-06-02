from torch.nn import Module
import torch
from torch.nn import functional as F

import torchsparse
import torchsparse.nn.functional as spf
import numpy as np
from sklearn.metrics import confusion_matrix


from MVRA.Voxel_Ops import Voxel_World
from MVRA.Features_2D import Backbone
from MVRA.Features_Sparse_3D import Unet_3d_Sparse
from MVRA.Attention import Multi_View_Attention
from MVRA.Heads import Twin_Head




def log_transform(x, shift=1):
          # https://github.com/magicleap/Atlas
          """rescales TSDF values to weight voxels near the surface more than close
          to the truncation distance"""
          return x.sign() * (1 + x.abs() / shift).log() 
def logits_wt(occuapncy):
  n_all = occuapncy.shape[0]
  n_p = occuapncy.sum()
  w_for_1 = (n_all - n_p).float() / n_p
  return w_for_1

class Multi_view_reconstruction(Module):
  def __init__(self,config,Attention=False):
    super().__init__()
    
    self.backbone=Backbone(config)    
    self.Unprojection = Voxel_World(config.voxel_size,config.voxel_dims,config.Scale_names)
    self.Refiner_coarse = Unet_3d_Sparse(config.Backbone_dim['coarse'],config.Unet3D_dim['coarse'])
    self.Refiner_middle = Unet_3d_Sparse(config.Backbone_dim['middle'],config.Unet3D_dim['middle'])


    if(Attention):
        self.use_attn = True
        self.attn = Multi_View_Attention(config.voxel_size,config.Backbone_dim['middle'],config.Unet3D_dim['middle'][-1])
        print("Using Attention mechanism")
    else:
      self.use_attn = False
    #self.Refiner_fine = Unet_3d_Sparse(config.Backbone_dim['fine'],config.Unet3D_dim['fine'])

    self.Head_coarse = Twin_Head(config.Unet3D_dim['coarse'][-1])
    self.Head_middle =   Twin_Head(config.Unet3D_dim['middle'][-1])
    #self.Head_fine =   Twin_Head(config.Unet3D_dim['fine'][-1])

    self.Scale_to_work = config.N_Scale_Model
    self.scales=config.N_Scale
    self.sparse=config.sparse_tensors
    self.teacherforce = True
    print("teacher forcing set to ",self.teacherforce)


    self.out_vol_count=1


  def loss_occ(self,pred_tsdf,pred_occ,ground):
    
    occupancy = ground.abs() < 0.999         
    loss = F.binary_cross_entropy_with_logits(pred_occ.float(), occupancy.float(), pos_weight=logits_wt(occupancy))
    
    return  loss

    
  def loss_tsdf(self,pred_tsdf,pred_occ,ground):
      
      occupancy = ground.abs() < 0.999

      #print(F.l1_loss(pred_tsdf[occupancy],ground[occupancy]))
      
      loss = F.binary_cross_entropy_with_logits(pred_occ.float(), occupancy.float(), pos_weight=logits_wt(occupancy))
      if(self.teacherforce):
        y = F.l1_loss(log_transform(pred_tsdf[occupancy]),log_transform(ground[occupancy])) 
      else:
        y = F.l1_loss(log_transform(pred_tsdf[pred_occ>0]),log_transform(ground[pred_occ>0]))

      

      return  y +loss

  def CreateSparseTensors(self,Tensor):

    # created to work only with 4 dimesions with feature dimension 1
    assert(len(Tensor.shape)==4)
    
    inds = torch.stack(torch.meshgrid( [torch.arange(0,i) for i in Tensor.shape],indexing='ij'),dim=-1).int() 
    inds = inds[:,:,:,:,[2,1,3,0]].to(Tensor.device)
    SparseTensor = torchsparse.SparseTensor(coords=inds.view(-1,4), feats=Tensor.view(-1,1))
    return SparseTensor
  def to_tsdf_volume(self,inds, vals):
    vol = torch.ones(240,240,192) 
    vol[inds[:, 1], inds[:, 0], inds[:, 2]] = vals.squeeze(-1).cpu()
    return vol.detach().numpy()
  def Realign(self,pred,gt):
    # align the coordinates
    pred_hash = spf.sphash(pred.C)
    gt_hash = spf.sphash(gt.C)
    idx_query = spf.sphashquery(pred_hash, gt_hash)
    if((idx_query != -1).all()==False):# Also handles duplicate coordinate
      raise Exception("check the coordinate values")

    return gt.F[idx_query]
 
  
  def forward(self,Input):
    
 
    
    # Loading data
    Features_scales=self.backbone.Handle_5(Input['images'])
    Projections=Input['projections']
    Origins=Input['vol_origin']
    Batch_size = Input['images'].shape[0]

    
  

    
    # creating initial Grids
    voxel_inds = self.Unprojection.world_grid
    voxel_inds = self.Unprojection.Add_batch(Batch_size,voxel_inds).to(Input['images'].device)
    Batch_inds = torch.arange(0,Batch_size+1) * self.Unprojection.world_size    
   
    curr_scale = 'coarse'
    ### coarse ###    
    Feature = Features_scales[0]
    tsdf_coarse = Input['tsdf_list'][2]
    Projections_coarse = Projections[:,:,2] 
    
    Voxel_Volume , voxel_mask = self.Unprojection.Back_project(Feature,Projections_coarse,Origins,curr_scale=curr_scale ,Coordinates=voxel_inds,Batch_indices=Batch_inds)    
    Voxel_Volume = torch.cat(Voxel_Volume,dim=0)
    Voxel_Volume_sparse = torchsparse.SparseTensor(coords=voxel_inds, feats=Voxel_Volume)
    Voxel_Volume_sparse = self.Refiner_coarse(Voxel_Volume_sparse)    
    
    Voxel_gt = self.CreateSparseTensors(tsdf_coarse) # converts to sparse tensor format by adding indices in X,Y,Z,Batch 
    Voxel_gt  = self.Realign(Voxel_Volume_sparse,Voxel_gt) # convert meshgrid order to Prediction's coordinates order 

    indices = Voxel_Volume_sparse.C
    Voxel_Volume_sparse = self.Head_coarse(Voxel_Volume_sparse.F)
    coarse_loss = self.loss_occ(*Voxel_Volume_sparse,Voxel_gt)

    # creating sparse grids from coarse tsdf
    #tsdf_coarse = Input['tsdf_list'][2]
    if(self.teacherforce):
      coarse_ground = self.CreateSparseTensors(tsdf_coarse)
      occupancy = coarse_ground.F.squeeze(1).abs() < 1.0
      voxel_inds = coarse_ground.C
    else:
      coarse_ground = Voxel_Volume_sparse[1] # score of being not 1
      occupancy = coarse_ground.squeeze(1) > 0
      voxel_inds = indices

    voxel_inds = voxel_inds[occupancy]
    if(int(len(voxel_inds))>8192*Batch_size):
      voxel_inds = voxel_inds[:8192*Batch_size]

    Batch_inds = [0]
    for i in range(Batch_size):
      Batch_inds.append(Batch_inds[-1] + sum(voxel_inds[:,-1]==i) * 8 ) 

    voxel_inds = self.Unprojection.Upsample_Coordinates(voxel_inds)

    
    curr_scale = 'middle'
    ### Middle ###  
    Feature = Features_scales[1] 
    tsdf_middle = Input['tsdf_list'][1]
    Projections_middle = Projections[:,:,1]     
    
    Voxel_Volume , voxel_mask = self.Unprojection.Back_project(Feature,Projections_middle,Origins,curr_scale=curr_scale ,Coordinates=voxel_inds,Batch_indices=Batch_inds)    
    Voxel_Volume = torch.cat(Voxel_Volume,dim=0)
    Voxel_Volume_sparse = torchsparse.SparseTensor(coords=voxel_inds, feats=Voxel_Volume)
    Voxel_Volume_sparse = self.Refiner_middle(Voxel_Volume_sparse)

    # Novel Attention    
    if(self.use_attn):
      Voxel_Volume_sparse = self.attn(Feature,Voxel_Volume_sparse,Projections_middle,Origins,curr_scale=curr_scale,count_mask=voxel_mask)
    
    Voxel_gt = self.CreateSparseTensors(tsdf_middle) # converts to sparse tensor format by adding indices in X,Y,Z,Batch 
    Voxel_gt  = self.Realign(Voxel_Volume_sparse,Voxel_gt) # convert meshgrid order to Prediction's coordinates order 
    Voxel_Volume_sparse = self.Head_middle(Voxel_Volume_sparse.F)
    fine_loss = self.loss_tsdf(*Voxel_Volume_sparse,Voxel_gt)
    





    """
    for i in range(8):
      tsdf_one =Voxel_Volume_sparse[0][Batch_inds[i]:Batch_inds[i+1]]
      tsdf_one_gt =Voxel_gt[Batch_inds[i]:Batch_inds[i+1]]
      occ = tsdf_one_gt<0.999
      tsdf_one = tsdf_one[occ]
      tsdf_one_gt = tsdf_one_gt[occ]
      tsdf_inds = voxel_inds[Batch_inds[i]:Batch_inds[i+1]][occ.squeeze(-1)].long()
      print(F.l1_loss(tsdf_one,tsdf_one_gt),i)
      import numpy as np
      import os
      np.savez_compressed(os.path.join('/root/Projectwork2/test_results', 'test_predtsdf_{}'.format(str(self.out_vol_count))), self.to_tsdf_volume(tsdf_inds,tsdf_one))
      np.savez_compressed(os.path.join('/root/Projectwork2/test_results', 'test_gttsdf_{}'.format(str(self.out_vol_count))), self.to_tsdf_volume(tsdf_inds,tsdf_one_gt))
      self.out_vol_count+=1
    #assert(1==2)"""


    
    return fine_loss+coarse_loss



    
    
    print("here")
