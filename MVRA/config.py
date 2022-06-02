import numpy as np

class MVRA_config(object):
  pass

config={
    
    'project_name':'MVRA',
    'strides':[4,8,16],
    'input_size':(640,480),
    'FPN_Dimension':64,
    'N_Scale' : 3,
    'sparse_tensors' : True ,
    'Scale_names':['fine','middle','coarse'],
    'voxel_size':0.04,
    'voxel_dims':[224, 224, 128],
    'N_Scale_Model':2,
    'Unet3D_dim':{'coarse': np.array([1, 2, 4, 3, 3])*32,
                  'middle'  : np.array([1, 2, 4, 3, 3])*24,
                  'fine'  : np.array([1, 2, 4, 3, 3])*16},
        
    'Backbone_dim':{'coarse': 80,
              'middle'  : 40,
              'fine'  : 24}


}
def get_config():
    cfg=MVRA_config()
    for k,v in config.items():
        setattr(cfg,k,v)
    return cfg

