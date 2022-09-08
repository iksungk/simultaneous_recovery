import torch

dtype = torch.cuda.FloatTensor
ctype = torch.complex64


def gradient_loss(img, weight, vmin = 0.075, vmax = 0.2):
    loss = torch.sum(torch.nn.ReLU()(vmin - torch.abs(img[:, :-1] - img[:, 1:])))
    loss += torch.sum(torch.nn.ReLU()(torch.abs(img[:, :-1] - img[:, 1:]) - vmax))
    loss += torch.sum(torch.nn.ReLU()(vmin - torch.abs(img[:-1, :] - img[1:, :])))
    loss += torch.sum(torch.nn.ReLU()(torch.abs(img[:-1, :] - img[1:, :]) - vmax))
    loss *= weight
    
    return loss


def laplacian_loss(img, weight, vmin = 0., vmax = 0.05):
    loss = torch.sum(torch.nn.ReLU()(vmin - torch.abs(img[1:, 1:] - img[:-1, 1:] - img[1:, :-1] + img[:-1, :-1])))
    loss += torch.sum(torch.nn.ReLU()(torch.abs(img[1:, 1:] - img[:-1, 1:] - img[1:, :-1] + img[:-1, :-1]) - vmax))
    loss *= weight
    
    return loss


def range_loss(img, weight, vmin = 0., vmax = 1.):
    return weight * torch.sum(torch.nn.ReLU()(vmin - img)) + torch.sum(torch.nn.ReLU()(img - vmax))
    

def wavelength_weights_loss(wv_g_gammas, weight_1, weight_2):
    loss = weight_1 * torch.square(torch.abs(1 - torch.sum(wv_g_gammas)))
    loss += weight_2 * torch.sum(torch.nn.ReLU()((-1) * wv_g_gammas))
    
    return loss


def wavelength_weights_loss_v2(wv_g_gammas, weight_1, weight_2):
    loss = weight_1 * torch.square(1 - torch.sum(wv_g_gammas ** 2))
    loss += weight_2 * torch.sum(torch.nn.ReLU()((-1) * wv_g_gammas))
    
    return loss


def tv_2d(img, tv_weight):
    h_variance = torch.sum(torch.abs(img[:, :-1] - img[:, 1:]))
    w_variance = torch.sum(torch.abs(img[:-1, :] - img[1:, :]))
    
    loss = tv_weight * (h_variance + w_variance)
    
    return loss


def fh_2d(img, fh_weight):
    hessian = torch.mean(torch.square(img[:-1, :-1] - img[:-1, 1:] - img[1:, :-1] + img[1:, 1:]))
    
    loss = fh_weight * hessian
    
    return loss


def npcc_loss(y_pred, y_true, weight):
    up = torch.mean((y_pred - torch.mean(y_pred)) * (y_true - torch.mean(y_true)))
    down = torch.std(y_pred) * torch.std(y_true)
    loss = 1 - up / down

    return weight * loss.type(dtype)