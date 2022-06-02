from torch.nn import Module
import torch
from torch.nn import functional as F

#  original version of Upsample from Vortx  https://github.com/noahstier/vortx
class Upsampler(Module):
    # nearest neighbor 2x upsampling for sparse 3D array

    def __init__(self):
        super().__init__()
        self.upsample_offsets = torch.nn.Parameter(
            torch.Tensor(
                [
                    [
                        [0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [1, 1, 0, 0],
                        [0, 1, 1, 0],
                        [1, 0, 1, 0],
                        [1, 1, 1, 0],
                    ]
                ]
            ).to(torch.int32),
            requires_grad=False,
        )
        self.upsample_mul = torch.nn.Parameter(
            torch.Tensor([[[2, 2, 2, 1]]]).to(torch.int32), requires_grad=False
        )

    def __call__(self, voxel_inds):
        return (
            voxel_inds[:, None] * self.upsample_mul + self.upsample_offsets
        ).reshape(-1, 4)


class Voxel_World(Module):
  # create a 3d space world sampled by voxel cubes of particular dimension

  def __init__(self,voxel_size,voxel_dims,scales):

    super().__init__()

    self.voxel_size = voxel_size
    self.dims = voxel_dims
    self.world_grid = None
    


    self.Interval={
        'coarse': 4,
        'middle': 2,
        'fine':  1,
    }

    self.Generate_grid('coarse')

    self.upsampler = Upsampler()



  def Generate_grid(self,curr_scale):
    
      with torch.no_grad():
        # interval higher for coarse cale lower for fine scale
        interval = self.Interval[curr_scale]
        # higher the interval voxel cube dimension is assumed to to be large because a large portion of a world(vocexl of corresponding scale) should concentrate on one pixel of small(coarse level) image
        x_range = torch.arange(0,self.dims[0]//interval)
        y_range = torch.arange(0,self.dims[1]//interval) 
        z_range = torch.arange(0,self.dims[2]//interval)


        # imaginary world with unit length  & voxels with size propotional to scale
        world_grid = torch.stack(torch.meshgrid(x_range, y_range,z_range,indexing='xy'),dim=-1).float()    # indexing 'xy' to keepvalue of x axis first and y second else it keeps in row column style  
        #print(world_grid.shape)
        #all points of the world (all axis) are made to single axis
        # shape -> no_voxels X coordinate point of each voxel(x,y,z)
        world_grid = world_grid.view(-1,3)     


        self.world_grid=world_grid
        self.world_size=world_grid.shape[0]



  def Add_batch(self,Batch_size,coords):
    # only for coarse
    batch_coords = []
    for B in range(Batch_size):
        batch_coords.append( torch.cat( [ torch.clone(coords) , torch.ones(self.world_size,1) * B]  , dim=1 ))
    batch_coords = torch.cat(batch_coords, dim=0).to(dtype=torch.int)
    return batch_coords
    
  def Upsample_Coordinates(self,Coordinates):

    return self.upsampler(Coordinates)

  def Sample(self,Features,voxel_coordinates,mask):# mask to handle NAN

      y = F.grid_sample(Features, voxel_coordinates, padding_mode='zeros', align_corners=True).squeeze(2)
      # remove nan(sometimes nan occurs)
      y[mask.unsqueeze(1).expand(-1, Features.shape[-3], -1) == False] = 0.0
      
      y = torch.sum(y,dim=0)

      return y


  def Back_project(self,Features,Projections,Origins,curr_scale=None,Coordinates=None,Batch_indices=None):

    # Features => features of image extracted from backbone
    # Projections => P= K [R | T ]
    # origin point (Centre point ) for each scene


    # size Batch X N_view  X ....


    # iterate through batches (Atlas,neuralrecon,Surfacenet)


    B = Features.shape[0]
    Batch_voxel_with_Features = []
    Batch_voxel_mask = []
    

    for b in range(B):

          #print(Coordinates.shape,Batch_indices)
          world_grid = Coordinates[Batch_indices[b]:Batch_indices[b+1]][:] 
          assert((world_grid[:,-1]==b).all())
          world_grid = world_grid[:,:-1] # removing batch number
          world_grid = world_grid * self.Interval[curr_scale]
          world_real = world_grid.float() * self.voxel_size + Origins[b].float()

          # X= [x,y,z,1] # z is for making sequential additive and multiplicative operations
          no_voxels=world_real.shape[0]
          world_real=torch.cat([world_real,torch.ones(no_voxels,1).to(world_real.device)],dim=1)


          # repeat for N views
          N= Features.shape[1]
          world_real=world_real.unsqueeze(0).expand(N,-1,-1)     
          
          feature_size,height, width =Features.shape[-3:]       
          proj = Projections[b]
          feats = Features[b]

          # Project the N view features to the N world grid
          # x = PX      

          voxel_with_Feature_coordinate = torch.bmm(proj , world_real.permute(0,2,1).contiguous())
          
          im_x, im_y, im_z = voxel_with_Feature_coordinate[:, 0], voxel_with_Feature_coordinate[:, 1], voxel_with_Feature_coordinate[:, 2]
          im_x = im_x / im_z
          im_y = im_y / im_z      
          
          
          # torch grid sample do not work like 0 to w-1 it works like -1 to 1 where -1 : 0 nd 1: w-1               
          voxel_with_Feature_coordinate = torch.stack([2 * im_x / (width - 1) - 1, 2 * im_y / (height - 1) - 1], dim=-1)

          #create mask
          #  original version of mask from NeuralRecon  https://github.com/zju3dv/NeuralRecon
          mask = voxel_with_Feature_coordinate.abs() <= 1 # ensuring its between 0 to h/w
          mask = (mask.sum(dim=-1) == 2) & (im_z > 0)  # ensuring both x , y , z  are proper
                    
          voxel_with_Feature_coordinate = voxel_with_Feature_coordinate.unsqueeze(1)
          
          voxel_with_Features =self.Sample(feats,voxel_with_Feature_coordinate,mask) #check afterwards
          voxel_mask = torch.sum(mask , dim=0)
          
          

          """
          print(torch.max(voxel_mask.reshape(-1)))
          hist = torch.histc(voxel_mask, bins = 28, min = 0, max = 27)
          print((hist/no_voxels)*100)"""
          

          Batch_voxel_mask.append(voxel_mask*1.0)
          
          voxel_mask[voxel_mask == 0] = 1 #to avoid inf
          voxel_with_Features /= voxel_mask.unsqueeze(0)#average

          Batch_voxel_with_Features.append(voxel_with_Features.permute(1, 0).contiguous())   #CN to NC   
          


    return Batch_voxel_with_Features ,torch.cat(Batch_voxel_mask,dim=0)
      