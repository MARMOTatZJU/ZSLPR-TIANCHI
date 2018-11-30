from torch.autograd import Function

# https://github.com/fungtion/DANN/blob/master/models/functions.py
# Usage : 
# from .reverse_utils import ReverseLayerF
# ReverseLayerF.apply(feature, alpha)

class ReverseLayerF(Function):

  @staticmethod
  def forward(ctx, x, alpha):
    ctx.alpha = alpha
    return x.view_as(x)

  @staticmethod
  def backward(ctx, grad_output):
    output = grad_output.neg() * ctx.alpha
    return output, None
