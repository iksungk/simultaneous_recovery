import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio

from tqdm import tqdm
from physical import forward_propagation
from losses import *


dtype = torch.cuda.FloatTensor

    
def plot_and_save_figure(nit, c_wv, displacements, obj, I_det, I_est,
                         one_iter_rec, loss_list, weights_list, Lz_list, meas_wv, meas_spec):
    x = (c_wv + displacements) * 1e9
    
    plt.figure(figsize = (20, 25))
    cm = 'gray'
    plt.subplot(5,4,1); plt.plot(np.abs(loss_list), label = "Loss"); plt.yscale('log'); plt.legend()
    plt.subplot(5,4,2); plt.plot(weights_list);
    plt.subplot(5,4,3); plt.plot(Lz_list);
    
    plt.subplot(5,4,4); plt.plot(x, weights_list[0] / np.max(weights_list[0]), label = 'Initial'); 
    plt.plot(x, weights_list[nit-1] / np.max(weights_list[nit-1]), label = 'Learned'); 
    plt.plot(meas_wv, meas_spec / np.max(meas_spec), label = 'Measured')
    plt.xlabel('Wavelength (nm)'); plt.ylabel('Relative weight'); plt.legend();
    
    
    plt.subplot(5,4,5); plt.imshow(one_iter_rec[0, 0, :, :], cmap = cm); plt.colorbar()
    plt.subplot(5,4,6); plt.imshow(one_iter_rec[1, 0, :, :], cmap = cm); plt.colorbar()
    plt.subplot(5,4,9); plt.imshow(np.abs(obj), cmap = cm); plt.colorbar()
    plt.subplot(5,4,10); plt.imshow(np.angle(obj), cmap = cm); plt.colorbar()
    plt.subplot(5,4,11); plt.imshow(I_det, cmap = cm); plt.colorbar()
    plt.subplot(5,4,12); plt.imshow(I_est, cmap = cm); plt.colorbar()
    plt.subplot(5,4,13); plt.imshow(one_iter_rec[0, 0, 128:384, 128:384], cmap = cm);
    plt.subplot(5,4,14); plt.imshow(np.abs(obj[128:384, 128:384]), cmap = cm);
    plt.subplot(5,4,15); plt.imshow(one_iter_rec[0, 0, 128:384, 128:384], cmap = cm);
    plt.subplot(5,4,16); plt.imshow(np.abs(obj[128:384, 128:384]), cmap = cm);
    
    m0 = np.abs(obj)
    m1 = np.abs(m0[:-1, :] - m0[1:, :])
    m2 = np.abs(m0[:, :-1] - m0[:, 1:])
    m3 = np.abs(m0[:-1, :-1] - m0[1:, :-1] - m0[:-1, 1:] + m0[1:, 1:])
    plt.subplot(5,4,18); plt.imshow(m1, cmap = cm); plt.colorbar()
    plt.subplot(5,4,19); plt.imshow(m2, cmap = cm); plt.colorbar()
    plt.subplot(5,4,20); plt.imshow(m3, cmap = cm); plt.colorbar()
    
    if nit % 50 == 0:
        plt.savefig('./figure/' + str(nit) + '_constant_' + str(len(displacements)) + '.png', bbox_inches = 'tight')
        
    plt.show()
    
    
def train(net_amp, 
          net_pha, 
          input_noise_net, 
          l_dz, 
          dim,
          uinc, 
          I_det, 
          one_iter_rec,
          p, 
          loss_list, 
          weights_list, 
          Lz_list,
          c_wv, 
          wv_g,
          meas_wv, 
          meas_spec,
          displacements, 
          lr, 
          epochs
         ):
    
    mse = torch.nn.MSELoss().type(dtype)
    
    params = list(net_amp.parameters())
    params += list(net_pha.parameters())
    params += list(wv_g.parameters())
    
    optimizer = torch.optim.Adam(params, lr=lr, betas = (0.9, 0.999))  # using ADAM opt
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, lr/100)

    bkg = uinc.cuda(0)
    I_det = I_det.cuda(0)
    
    for i in tqdm(range(1, 1 + epochs)):
        optimizer.zero_grad()
        for param in params:
            param.grad = None
        
        net_input = input_noise_net()
        amp = net_amp(net_input).squeeze(0).squeeze(0)
        pha = net_pha(net_input).squeeze(0).squeeze(0)
        obj = amp * torch.exp(1j * pha)
    
        Lz_list[i-1] = l_dz
        
        intensities = forward_propagation(obj, bkg, p, displacements, l_dz, c_wv = c_wv, obj_dim = dim)
        I_est = wv_g(intensities)

        total_loss = mse(I_est, I_det)
        total_loss += npcc_loss(I_est, I_det, 1e1)
        total_loss += wavelength_weights_loss(wv_g.gammas, 1e1, 2.5e-1)
        total_loss += tv_2d(amp, 0.5e-4) 
        total_loss += tv_2d(pha, 0.5e-4) 
        
        if i % 50 == 0:
            print('Epoch ' + str(i) + ': '  + str(torch.sum(wv_g.gammas ** 2)))
        
        loss_list[i-1] = total_loss.cpu().detach().numpy()
        weights_list[i-1] = wv_g.gammas.cpu().detach().numpy()
        
        total_loss.backward()

        optimizer.step()
        scheduler.step()
        
        
        if (i <= 50 and i % 10 == 0) or (i > 50 and i <= 300 and i % 25 == 0) or (i > 300 and i % 50 == 0):
                plot_and_save_figure(i, c_wv, displacements,
                                     obj.cpu().detach().numpy(), 
                                     I_det.cpu().detach().numpy(),
                                     I_est.cpu().detach().numpy(), 
                                     one_iter_rec,
                                     loss_list,
                                     weights_list,
                                     Lz_list,
                                     meas_wv,
                                     meas_spec)
                
            
        if i % 50 == 0:
            sio.savemat('./rec/' + str(i) + '_constant_' + str(len(displacements)) + '.mat', 
                        mdict = {'obj':obj.cpu().detach().numpy(),
                                 'I_det':I_det.cpu().detach().numpy(),
                                 'I_est':I_est.cpu().detach().numpy(),
                                 'loss_list':loss_list,
                                 'weights_list':weights_list,
                                 'Lz_list':Lz_list})