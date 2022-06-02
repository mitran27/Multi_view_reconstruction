import torchvision
import torch
from torch.nn import Module,Sequential,Conv2d,ModuleList,BatchNorm2d,ReLU,Linear
from torch.nn import functional as F


class Convolutional_Block(Module):
  def __init__(self,indim,outdim,kernel,pad=0):

    super().__init__()

    self.conv = Conv2d(in_channels=indim, out_channels=outdim, kernel_size=kernel,padding=pad,bias=False)
    self.bn = BatchNorm2d(outdim)
    self.relu = ReLU()


  def forward(self, x):
     x = self.conv(x)
     x = self.bn(x)
     x = self.relu(x)
     return x



class Backbone(Module):

  # returnFeatures with  i/4 ,i/8 ,i/16 scale

  def __init__(self,config):
         
          super().__init__()

          model=torchvision.models.mnasnet1_0(pretrained=True, progress=True)
          self.Feature_Extractor = ModuleList()
          
          # extract featues at strid 4,8,16 which are at 8,9,10 correspondingly for mnasnet
          self.Feature_Extractor.append(Sequential( *[model.layers._modules[str(i)] for i in range(0,9)]))
          self.Feature_Extractor.append(model.layers._modules[str(9)])
          self.Feature_Extractor.append(model.layers._modules[str(10)])

          
          for i in range(3):
            for param in self.Feature_Extractor[i].parameters():
                param.requires_grad = False

          mnas_dims=[24,40,80]
          self.mnas=mnas_dims
          fpn=config.FPN_Dimension
          
          
          self.Conv=ModuleList([
                                Conv2d(mnas_dims[0],fpn ,1,padding=0,bias=False),
                                Conv2d(mnas_dims[1],mnas_dims[2],1,padding=0,bias=False),
                                Conv2d(mnas_dims[2],mnas_dims[2],1,padding=0,bias=False)
          ])
          for C in range(3):
            torch.nn.init.kaiming_normal_(self.Conv[C].weight) 


          self.Out=ModuleList([
                                Convolutional_Block(mnas_dims[2],mnas_dims[1],3,1),
                                Convolutional_Block(fpn,mnas_dims[0],3,1)
          ])       


  def Handle_5(self,X):
      B,N,C,H,W =X.shape
      D=self.mnas
      X=X.view(B*N,C,H,W)
      
      coarse,middle = self.forward(X)
     
      return [coarse.view(B,N,D[2],H//16,W//16),
              middle.view(B,N,D[1],H//8,W//8)
              ]
    
  def forward(self,X):
     
      
        
      output=[]
  
      y1=self.Feature_Extractor[0](X)
      y2=self.Feature_Extractor[1](y1)
      y3=self.Feature_Extractor[2](y2) 

      y1=self.Conv[0](y1)
      y2=self.Conv[1](y2)
      y3=self.Conv[2](y3)
      output.append(y3)

      y2=F.interpolate(y3,size=y2.shape[-2:])+y2
      y2=self.Out[0](y2)
      output.append(y2)
      """
      y1=F.interpolate(y2,size=y1.shape[-2:])+y1
      y1_out=self.Out[1](y1)
      output.append(y1_out)"""

      return output