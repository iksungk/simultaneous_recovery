import torch
import torch.nn as nn

dtype = torch.cuda.FloatTensor
ctype = torch.complex64


class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 'same', bias = True)
        self.bn = nn.BatchNorm2d(out_channels, affine = True)
        self.act = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        
        return x

    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = (2, 2)):
        super(DownBlock, self).__init__()
        
        if stride == (2, 2):
            padding = 1
        else:
            padding = 'same'
            
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, 
                               padding = padding, bias = True)
        self.bn1 = nn.BatchNorm2d(out_channels, affine = True)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 'same', bias = True)
        self.bn2 = nn.BatchNorm2d(out_channels, affine = True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        return x
    
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor = (2, 2)):
        super(UpBlock, self).__init__()
        
        self.up = nn.Upsample(scale_factor = scale_factor, mode = 'nearest', align_corners = None)
        self.bn0 = nn.BatchNorm2d(in_channels, affine = True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 'same', bias = True)
        self.bn1 = nn.BatchNorm2d(out_channels, affine = True)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 1, stride = 1, padding = 'same', bias = True)
        self.bn2 = nn.BatchNorm2d(out_channels, affine = True)

    def forward(self, x, skip): 
        x = self.up(x)     
        if skip is not None:
            x = torch.cat((x, skip), dim = 1)
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
                
        return x
    
    
class SharedEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, channels_down, 
                 skip_channels = 8):
        super(SharedEncoder, self).__init__()
        
        self.down1 = DownBlock(in_channels, channels_down[0], stride = (2, 2))
        self.down2 = DownBlock(channels_down[0], channels_down[1], stride = (2, 2))
        self.down3 = DownBlock(channels_down[1], channels_down[2], stride = (2, 2))
        self.down4 = DownBlock(channels_down[2], channels_down[3], stride = (2, 2))
        self.down5 = DownBlock(channels_down[3], channels_down[4], stride = (2, 2))

        self.skip1 = SkipBlock(in_channels, skip_channels)
        self.skip2 = SkipBlock(channels_down[0], skip_channels)
        self.skip3 = SkipBlock(channels_down[1], skip_channels)
        self.skip4 = SkipBlock(channels_down[2], skip_channels)
        self.skip5 = SkipBlock(channels_down[3], skip_channels)
                               
    def forward(self, x):
        s1 = self.skip1(x)
        x = self.down1(x)
        s2 = self.skip2(x)
        x = self.down2(x)
        s3 = self.skip3(x)
        x = self.down3(x)
        s4 = self.skip4(x)
        x = self.down4(x)
        s5 = self.skip5(x)
        x = self.down5(x)

        return x, s1, s2, s3, s4, s5
    
    
class DecoderBranch(nn.Module):
    def __init__(self, in_channels, out_channels, channels_down, channels_up, sigmoid_min, sigmoid_max, 
                 skip_channels = 8):
        super(DecoderBranch, self).__init__()
        
        self.sigmoid_min = sigmoid_min
        self.sigmoid_max = sigmoid_max

        self.up1 = UpBlock(channels_down[4] + skip_channels, channels_up[4], scale_factor = (2, 2))
        self.up2 = UpBlock(channels_up[4] + skip_channels, channels_up[3], scale_factor = (2, 2))
        self.up3 = UpBlock(channels_up[3] + skip_channels, channels_up[2], scale_factor = (2, 2))
        self.up4 = UpBlock(channels_up[2] + skip_channels, channels_up[1], scale_factor = (2, 2))
        self.up5 = UpBlock(channels_up[1] + skip_channels, channels_up[0], scale_factor = (2, 2))
        
        self.conv = nn.Conv2d(channels_up[0], out_channels, kernel_size = 1, stride = 1, padding = 'same', bias = True)
        
    def forward(self, x, s1, s2, s3, s4, s5):
        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)
        
        x = self.conv(x)
        
        x = torch.sigmoid(x) * (self.sigmoid_max - self.sigmoid_min) + self.sigmoid_min
        
        return x
    
    
class model_parallel(nn.Module):
    def __init__(self, in_channels, out_channels, channels_down, channels_up, vmin, vmax, device_num = 0):
        super(model_parallel, self).__init__()
                
        self.device_num = device_num
        
        self.enc = SharedEncoder(in_channels, out_channels, channels_down).type(dtype).cuda(self.device_num)
        
        self.dec = DecoderBranch(in_channels, out_channels, channels_down, channels_up, 
                                     sigmoid_min = vmin, sigmoid_max = vmax).type(dtype).cuda(self.device_num)      

    def forward(self, z):
        z = z.cuda(self.device_num)
        q, s1, s2, s3, s4, s5 = self.enc(z)
        x = self.dec(q, s1, s2, s3, s4, s5)

        return x
    
    
class noise_net(nn.Module):
    def __init__(self, input_dims):
        super(noise_net, self).__init__()
        self.input_dims = input_dims
        self.input_noise = torch.nn.Parameter(0.1 * torch.rand(self.input_dims), requires_grad = True).type(dtype)
        
    def forward(self):
        return self.input_noise
    
    
def weights_init_amp(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data, gain = 0.3)
        
        if m.bias is not None:
             nn.init.constant_(m.bias.data, 0)
                
    elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
        
        
def weights_init_pha(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data, gain = 0.3)
        
        if m.bias is not None:
             nn.init.constant_(m.bias.data, 0)
                
    elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)