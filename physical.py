import torch
import torch.nn as nn
import torch.nn.functional as F

dtype = torch.cuda.FloatTensor
ctype = torch.complex64


class PhysicalSystem():
    def __init__(self, dim):
        super(PhysicalSystem, self).__init__()
        self.dim = dim
        
    def initialize_parameters(self, c_wv, dz, wv_displacement):
        pi = torch.acos(torch.zeros(1)).item() * 2
        
        self.lb = c_wv + wv_displacement
        self.c = 299792458
        self.k0 = 2 * pi / self.lb
        self.n0 = 1
        self.k = self.n0 * self.k0
        
        self.Nx = self.dim
        self.Ny = self.Nx
        dx = 3.8e-6
        dy = dx
        self.dz = dz
        
        self.Lx = self.Nx * dx
        self.Ly = self.Ny * dy
        
        x = dx * torch.arange(-self.Nx//2, self.Nx//2)
        y = dy * torch.arange(-self.Ny//2, self.Ny//2)
        X, Y = torch.meshgrid(x, y)
        self.X2 = torch.add(torch.square(X), torch.square(Y))
        
        dkx = 2 * pi / self.Lx
        dky = 2 * pi / self.Ly
        kx = dkx * torch.arange(-self.Nx//2, self.Nx//2)
        ky = dky * torch.arange(-self.Ny//2, self.Ny//2)
        Kx, Ky = torch.meshgrid(kx, ky)
        self.K2 = torch.add(torch.square(Kx), torch.square(Ky))
        
        
    def forward_operator(self):
        return torch.fft.fftshift(torch.exp(-1j * (self.k - torch.real(torch.sqrt(self.k ** 2 - self.K2).type(ctype))) * self.dz))
        
        
    def inverse_operator(self):
        return torch.fft.fftshift(torch.exp(1j * (self.k - torch.real(torch.sqrt(self.k ** 2 - self.K2).type(ctype))) * self.dz))
    
    
    def planar_illumination(self, width = 1):
        sig = width * self.Lx
        return torch.exp(-self.X2 / 2 / sig ** 2).type(torch.FloatTensor)
    
        
def to_phasor_notation(amp, pha, device_num):
    if device_num == 'cpu':
        return amp.cpu() * torch.exp(1j * pha.cpu())
    else:
        return amp.cuda(device_num) * torch.exp(1j * pha.cuda(device_num))


def pad_object(obj, pad_size):
    return F.pad(obj, [pad_size, pad_size, pad_size, pad_size])


def forward_propagation(obj, bkg, p, displacements, dz, c_wv, obj_dim = 512):
    intensities = []
    
    for displacement in displacements:
        p.initialize_parameters(c_wv, dz, displacement)
        assert p.Nx == obj_dim, "Check dimensions of a computational grid!"
        
        H = p.forward_operator().cuda(0)
        H_inv = p.inverse_operator().cuda(0)
        
        uinc = bkg.cuda(0)
#         uinc = bkg[800-256 : 800+256, 800-256 : 800+256].cuda(0)
        uinc = torch.fft.ifft2(torch.fft.fft2(uinc) * H_inv)
        assert uinc.shape == (obj_dim, obj_dim), "Check dimensions of an illumination profile!"

        psi = torch.fft.ifft2(torch.fft.fft2(obj * uinc) * H)
        
        intensities.append(torch.square(torch.abs(psi)))
    
    return torch.stack(intensities)


class wavelength_gammas(nn.Module):
    def __init__(self, num, mode, mu = 810e-9, std = 30e-9, left_width = 100e-9, right_width = 100e-9, exp_x = None, exp_y = None):
        super(wavelength_gammas, self).__init__()
        
        pi = torch.acos(torch.zeros(1)).item() * 2
        
        if mode == 'constant':
            self.gammas = torch.nn.Parameter(torch.full((int(num), ), 1/num, device='cuda:0'))

        elif mode == 'gaussian':
            self.x = torch.linspace(-left_width, right_width, num, device = 'cuda:0', requires_grad = False) + mu
            gammas = 1 / np.sqrt(2 * pi) / std * torch.exp(-0.5 * torch.square((self.x - mu) / std))
            gammas /= torch.sum(gammas)
            self.gammas = torch.nn.Parameter(gammas, requires_grad = True)
            
        elif mode == 'random':
            gammas = torch.rand((int(num), ), device='cuda:0')
            gammas /= torch.sum(gammas)
            self.gammas = torch.nn.Parameter(gammas)

        elif mode == 'experiment':
            self.x = exp_x * 1e-9
            gammas = torch.from_numpy(exp_y).cuda(0)
            gammas /= torch.sum(gammas)
            self.gammas = torch.nn.Parameter(gammas, requires_grad = True)
            
            
    def forward(self, inp):
        return torch.sum(inp * (self.gammas.unsqueeze(-1).unsqueeze(-1)), dim = 0)