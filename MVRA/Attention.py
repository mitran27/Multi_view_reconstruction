from MVRA.Features_Sparse_3D import Sparse_3d_Residual_Block
from torch.nn import Module,Linear,Conv2d
import torch
from torch.nn import functional as F
import torchsparse

class Multi_View_Attention(Module):
  # create a 3d space world sampled by voxel cubes of particular dimension

  def __init__(self,voxel_size,backbone_dim,Dimension):

    super().__init__()
    self.layer1 = Multi_View_Attention_block(voxel_size,backbone_dim,Dimension)
    self.layer2 = Multi_View_Attention_block(voxel_size,backbone_dim,Dimension)
  def forward(self,Image_Features,Voxel_Sparse,Projections,Origins,curr_scale=None,count_mask=None):
    y = self.layer1.forward(Image_Features,Voxel_Sparse,Projections,Origins,curr_scale,count_mask)
    y = self.layer2.forward(Image_Features,y,Projections,Origins,curr_scale,count_mask)
    return y

class Multi_View_Attention_block(Module):
  # create a 3d space world sampled by voxel cubes of particular dimension

  def __init__(self,voxel_size,backbone_dim,Dimension):

    super().__init__()

    self.voxel_size = 0.04
    self.D = Dimension
    self.W_query = Linear(Dimension, Dimension)
    self.W_key = Conv2d(backbone_dim, Dimension,1)  
    self.Refine =  Linear(Dimension, Dimension)
    self.view_thresh =1

    self.Refine_3D = Sparse_3d_Residual_Block(Dimension, Dimension, 3)



    self.Interval={
        'coarse': 4,
        'middle': 2,
        'fine':  1,
    }

  
  



  def Attention_Sample(self,key_Features,voxel_coordinates,mask,query_features):# mask to handle NAN

      Dimension = key_Features.shape[-3]
      key_feats = F.grid_sample(key_Features, voxel_coordinates, padding_mode='zeros', align_corners=True).squeeze(2)
      key_feats[mask.unsqueeze(1).expand(-1,Dimension,-1) == False] = 0.0

      # Sample x seqlen x Dimension
      query_feats = query_features.unsqueeze(1)
      key_feats = key_feats.permute(2,0,1)
      

      # ATTENTION
      # 1) Matmul query and key (scale if needed)
      # 2) score sotfmax 
      # 3) Multiply key and score


      scale = self.D ** 0.5
      attn_weights = torch.matmul(query_feats,key_feats.transpose(1, 2)) / scale
      attn_weights[mask.transpose(1,0).unsqueeze(1)==False] = -10000.0
      attn_probs = F.softmax(attn_weights,dim=-1)
      y = torch.matmul(attn_probs, key_feats).squeeze(1)

      return y


  def forward(self,Image_Features,Voxel_Sparse,Projections,Origins,curr_scale=None,count_mask=None):

    # Features => features of image extracted from backbone
    # Projections => P= K [R | T ]
    # origin point (Centre point ) for each scene
    # size Batch X N_view  X ....
    # iterate through batches (Atlas,neuralrecon,Surfacenet)
    # Voxel_Sparse must be  sparse type
    #print(Image_Features.shape,Voxel_Sparse.C.shape,Voxel_Sparse.F.shape,Projections.shape,curr_scale,count_mask.shape)
    

    B,N = Image_Features.shape[:2]
    Batch_voxel_with_Features = []
    Batch_voxel_mask = []
    Coordinates = Voxel_Sparse.C
    Voxel_Features = Voxel_Sparse.F

    for b in range(B):


          # create initial world coordinates for the batch with view grater than thresh
          world_grid = Coordinates[(count_mask >1) & (Coordinates[:,-1]==b)]
          assert((world_grid[:,-1]==b).all())
          world_grid = world_grid[:,:-1] # removing batch number
          world_grid = world_grid * self.Interval[curr_scale]
          world_real = world_grid.float() * self.voxel_size + Origins[b].float()
         

          # X= [x,y,z,1] # z is for making sequential additive and multiplicative operations
          no_voxels=world_real.shape[0]
          world_real=torch.cat([world_real,torch.ones(no_voxels,1).to(world_real.device)],dim=1)
          # repeat for N views          
          world_real=world_real.unsqueeze(0).expand(N,-1,-1)              
          feature_size=Image_Features.shape[-3]                  
            
          # read 
          proj = Projections[b]
          img_feats =  Image_Features[b]
          vox_feats = Voxel_Features[(count_mask >1) & (Coordinates[:,-1]==b)]
          height, width = img_feats.shape[-2:]       

          # Project the N view features to the N world grid
          # x = PX      
          voxel_with_Feature_coordinate = torch.bmm(proj , world_real.permute(0,2,1).contiguous())          
          im_x, im_y, im_z = voxel_with_Feature_coordinate[:, 0], voxel_with_Feature_coordinate[:, 1], voxel_with_Feature_coordinate[:, 2]
          im_x = im_x / im_z
          im_y = im_y / im_z            
          # torch grid sample do not work like 0 to w-1 it works like -1 to 1 where -1 : 0 nd 1: w-1         
          voxel_with_Feature_coordinate = torch.stack([2 * im_x / (width - 1) - 1, 2 * im_y / (height - 1) - 1], dim=-1)   
          
          mask = voxel_with_Feature_coordinate.abs() <= 1 # ensuring its between 0 to h/w
          mask = (mask.sum(dim=-1) == 2) & (im_z > 0)  # ensuring both x , y , z  are proper
                  
         

          # Multiply with the weights
          img_feats = self.W_key(img_feats)
          vox_feats = self.W_query(vox_feats) 

          voxel_with_Feature_coordinate = voxel_with_Feature_coordinate.unsqueeze(1)     
          y = self.Attention_Sample(img_feats,voxel_with_Feature_coordinate,mask,vox_feats) 

          #Voxel_Features[(count_mask >1) & (Coordinates[:,-1]==b)] = Voxel_Features[(count_mask >1) & (Coordinates[:,-1]==b)] + y  // inplace error
          Voxel_Features_tmp = torch.zeros_like(Voxel_Features)
          Voxel_Features_tmp[(count_mask >1) & (Coordinates[:,-1]==b)] = y 
          Voxel_Features = Voxel_Features + Voxel_Features_tmp


    Voxel_Sparse = torchsparse.SparseTensor(coords=Coordinates, feats=Voxel_Features)
    
    Voxel_Sparse = self.Refine_3D(Voxel_Sparse)

    return Voxel_Sparse
          


          


      