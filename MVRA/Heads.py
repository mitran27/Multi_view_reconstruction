from torch.nn import Linear
from torch.nn import Module


class Twin_Head(Module):
  def __init__(self,Dimension):
    super().__init__()

    self.tsdf_linear = Linear(Dimension, 1)
    self.occ_linear = Linear(Dimension, 1)

  def forward(self,X):
    tsdf = self.tsdf_linear(X)
    occ = self.occ_linear(X)

    return tsdf,occ