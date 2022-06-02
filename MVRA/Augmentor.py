import torch
import numpy as np
class Augment_3D:
    def __init__(self, voxel_size,paddingXY=1.5, paddingZ=.25):
        self.padding_start = torch.tensor([paddingXY, paddingXY, paddingZ])
        self.padding_end = torch.tensor([paddingXY, paddingXY, 0])
        self.voxel_size = voxel_size
    def get_transform_matrix(self,r,volumeshape,voxel_size,origin):

        R = torch.tensor([[np.cos(r), -np.sin(r)],[np.sin(r), np.cos(r)]], dtype=torch.float32)
        T = torch.eye(4)
        T[:2, :2] = R

        return T
    def coordinates(self,voxel_dim):   

        nx, ny, nz = voxel_dim
        x = torch.arange(nx, dtype=torch.long)
        y = torch.arange(ny, dtype=torch.long)
        z = torch.arange(nz, dtype=torch.long)
        x, y, z = torch.meshgrid(x, y, z,indexing ='ij')
        return torch.stack((x.flatten(), y.flatten(), z.flatten()))
    def transform(self,tsdfs,old_origin,new_origin,Transform):

        x,y,z = tsdfs[0].shape
        coords = self.coordinates(tsdfs[0].shape)
        world = coords.type(torch.float) * self.voxel_size + new_origin.view(3, 1)
        world = torch.cat((world, torch.ones_like(world[:1])), dim=0)
        world = Transform[:3, :] @ world
        coords = (world - old_origin.view(3, 1)) / self.voxel_size

        tsdf_list = []

        for l, tsdf_s in enumerate(tsdfs):

            coords_world_s = coords.view(3, x, y, z)[:, ::2 ** l, ::2 ** l, ::2 ** l] / 2 ** l
            dim_s = list(coords_world_s.shape[1:])
            coords_world_s = coords_world_s.view(3, -1)

            old_voxel_dim = list(tsdf_s.shape)

            coords_world_s = 2 * coords_world_s / (torch.Tensor(old_voxel_dim) - 1).view(3, 1) - 1
            coords_world_s = coords_world_s[[2, 1, 0]].T.view([1] + dim_s + [3])

            tsdf_vol = torch.nn.functional.grid_sample(
                    tsdf_s.view([1, 1] + old_voxel_dim),
                    coords_world_s, mode='nearest', align_corners=False
            ).squeeze()
            tsdf_vol_bilin = torch.nn.functional.grid_sample(
                tsdf_s.view([1, 1] + old_voxel_dim), coords_world_s, mode='bilinear',
                align_corners=False
            ).squeeze()
            mask = tsdf_vol.abs() < 1
            tsdf_vol[mask] = tsdf_vol_bilin[mask]

            # padding_mode='ones' does not exist for grid_sample so replace
            # elements that were on the boarder with 1.
            # voxels beyond full volume (prior to croping) should be marked as empty
            mask = (coords_world_s.abs() >= 1).squeeze(0).any(3)
            tsdf_vol[mask] = 1

            tsdf_list.append(tsdf_vol)

        return tsdf_list



    def __call__(self,tsdfs,voxel_info,R_t):

        origin = voxel_info
        r = torch.rand(1) * 2*np.pi
        T = self.get_transform_matrix(r,tsdfs[0].shape,self.voxel_size,origin)
       
        
        for i in range(len(R_t)):
            R_t[i] = T.numpy() @ R_t[i]


        new_origin = origin + T[:3,-1]
        tsdf_list = self.transform(tsdfs,origin,new_origin, T.inverse())
        

        return  tsdf_list,new_origin,R_t

        

        




