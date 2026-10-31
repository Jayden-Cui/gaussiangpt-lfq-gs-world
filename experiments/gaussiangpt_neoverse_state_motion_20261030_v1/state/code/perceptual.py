"""Official five VGG19 feature slices, one shared model allocation."""
import torch
from torch import nn
import torch.nn.functional as F


class Vgg19(nn.Module):
    def __init__(self, cache):
        super().__init__()
        import torchvision
        torch.hub.set_dir(str(cache))
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT)
        self.blocks = nn.ModuleList([vgg.features[a:b] for a, b in [(0,4), (4,9), (9,18), (18,27), (27,36)]])
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])
        self.requires_grad_(False).eval()

    def forward(self, image, target):
        x = (image.permute(2, 0, 1)[None]-self.mean)/self.std
        y = (target.permute(2, 0, 1)[None]-self.mean)/self.std
        x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
        y = F.interpolate(y, size=(224,224), mode='bilinear', align_corners=False)
        loss = x.new_zeros(())
        for block in self.blocks:
            x = block(x)
            with torch.no_grad():
                y = block(y)
            loss = loss + F.mse_loss(x, y)
        return loss


def ssim(image, target):
    """RGB SSIM, Gaussian 11x11 sigma=1.5, data_range=1, valid borders."""
    x, y = image.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
    axis = torch.arange(11, device=x.device, dtype=x.dtype)-5
    g = torch.exp(-axis.square()/(2*1.5**2))
    g = g/g.sum()
    kernel = (g[:, None]*g[None, :])[None, None].expand(3, 1, 11, 11)
    def blur(v):
        return F.conv2d(v, kernel, groups=3)
    ux, uy = blur(x), blur(y)
    vx, vy, vxy = blur(x*x)-ux*ux, blur(y*y)-uy*uy, blur(x*y)-ux*uy
    return float((((2*ux*uy+.01**2)*(2*vxy+.03**2))/
                  ((ux.square()+uy.square()+.01**2)*(vx+vy+.03**2))).mean())
