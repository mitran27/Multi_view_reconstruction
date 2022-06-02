import numpy as np 
import torch

import pickle
import cv2
from os import listdir

from MVRA.Augmentor import Augment_3D

def Projection_Matrix(intrinsic,extrinsic,strides):


  # P= k[R|t]

  # scannet given camera to world which is inverse of [R|t]
  #projections for all images in all scales in the features


  Projections_list=[]
  stride=4
  for i in range(len(extrinsic)):
    
      R_t = np.linalg.inv(extrinsic[i])
      K   = intrinsic[i]
      Projections=[]
      for s in range(3):
          
          K_s =np.eye(3)
          K_s[:2,:] = K[:2,:] / stride / 2 ** s
          P  = K_s @ R_t[:3, :4]
          
          Projections.append(P)
      Projections_list.append(Projections)

  return np.array(Projections_list)     

class ScannetDataset():
  def __init__(self,scene_path,tsdf_path,config,augment=False):

    assert len(listdir(scene_path))==len(listdir(scene_path))
    
    
    self.scene_pth=scene_path
    self.tdsf_pth=tsdf_path
    self.scenes=sorted(listdir(scene_path))
    self.strides=config.strides
    self.scale=len(config.strides)
    self.cfg=config
    self.augment = augment
    self.augmentor = Augment_3D(config.voxel_size)

    print("files loaded")
  def __len__(self):
    return len(self.scenes)
    

    
  def normalize(self,image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = image.astype(np.float32)
    image /= 255.
    image -= mean
    image /= std
    return image
  def resize(self,new_dims,image_list,intrinsic):

    intrinsics_list=[]
    resized_images=[]
    h,w=image_list[0].shape[:2]
    size=new_dims
    for i, im in enumerate(image_list):
          img = cv2.resize(im,size, cv2.INTER_LINEAR )
          resized_images.append(np.array(img, dtype=np.float32))


    K=intrinsic.copy()
    K[0, :] /= (w / size[0])
    K[1, :] /= (h / size[1])

    intrinsics_list = [K*1.0  for _ in range(len(resized_images))]     


    return np.array(resized_images),intrinsics_list

  def read_images(self,path):

    image_list=[]
    for i in range(len(listdir(path))):

      img_name=str(i)+'.jpg'
      img=cv2.imread(path+img_name)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = self.normalize(img)

      # preprocess
      
      image_list.append(img)

    return image_list    


  def read_depths(self,path):

    depth_list=[]
    for i in range(len(listdir(path))):

      dep_name=str(i)+'.png'
      depth = cv2.imread(path+dep_name, -1).astype(np.float32)
      depth /= 1000.0  
      depth[depth > 3.0] = 0

      depth_list.append(depth)

    return depth_list


  def read_camera_matrix(self,path):

    # read K for the scene and [R|T] for all images
    # reading pose 

    extrinsics=[]
    for i in range(len(listdir(path+'pose'))):
      cam_pose = np.loadtxt( path+'pose/'+str(i) + ".txt", delimiter=' ')
      extrinsics.append(cam_pose)

    # reading K
    intrinsics = np.loadtxt(path+ 'intrinsic/'+ 'intrinsic_color.txt', delimiter=' ')[:3, :3]

    return intrinsics,extrinsics
  def make_volume_newshape(self,volume,newshape):
  
        y,x,z = volume.shape
        ny = min(y,newshape[0])
        nx = min(x,newshape[1])
        nz = min(z,newshape[2])
        cat_volume = volume[:ny , :nx , :nz]
        new_volume = np.ones(newshape,dtype=np.float)
        new_volume[:ny,:nx,:nz] = cat_volume

        return new_volume

  def read_tsdf(self,path):
    
    # implement lrucache :todo
    shape={
        0:[224,224,128],
        1:[112,112,64],
        2:[56,56,32]
    }

    tsdf_list=[]

    for i in range(self.scale):

      tsdf = np.load(path+ 'full_tsdf_layer{}.npz'.format(i),allow_pickle=True)
      tsdf = tsdf.f.arr_0
      tsdf = self.make_volume_newshape(tsdf,shape[i])
      tsdf_list.append(tsdf)

    with open(path+'tsdf_info.pkl', 'rb') as f:
            voxel_info = pickle.load(f)['vol_origin']
    #voxel_info=np.loadtxt(path+ 'voxel_origin.txt', delimiter=' ')

    return tsdf_list,voxel_info

  def __getitem__(self,idx):

    curr_scene=self.scenes[idx]
    tsdf_list,voxel_info=self.read_tsdf(self.tdsf_pth+curr_scene+'/')

    images=self.read_images(self.scene_pth+curr_scene+'/color'+'/')
    #depths=self.read_depths(self.scene_pth+curr_scene+'/depth'+'/')

    K,R_t =self.read_camera_matrix(self.scene_pth+curr_scene+'/')
    images,K_list=self.resize(self.cfg.input_size,images,K)

    tsdf_list=[torch.tensor(tsdf,dtype=torch.float32) for tsdf in tsdf_list]
    voxel_info = torch.tensor(voxel_info,dtype=torch.float32)

    if(self.augment and torch.rand(1)>0.6):
       #print("augment")
       tsdf_list,voxel_info,R_t = self.augmentor(tsdf_list,voxel_info,R_t)

    Projections_list=Projection_Matrix(K_list,R_t,self.strides)

    

    images = torch.permute(torch.tensor(images,dtype=torch.float32),(0,3,1,2))

    z= {
        
            'images':images,            
            'tsdf_list': tsdf_list,
            'projections' : torch.tensor(Projections_list,dtype=torch.float32),
            'vol_origin': voxel_info}
    return z
         
