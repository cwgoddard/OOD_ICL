import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import beta
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize

def sample_cone(n, d, max_theta, min_theta=0.0, r=1.0, gaussianize=False, device='cuda'):
    def sample_sphere_(n, d, r=1.0):
        if type(r) is not np.ndarray:
            r = np.ones(n) * r
        u = np.random.normal(size=(n, d))
        norms = np.sqrt(np.sum(u**2, axis=1))
        return r[:, None] * u / norms[:, None]

    #sample the first coordinate of the vectors
    d_sphere = d-1 #S^(d-1) is embedded in d-dim'l space
    rho = np.sqrt(2*(r**2)*(1-np.cos(max_theta)))
    rho_min = np.sqrt(2*(r**2)*(1-np.cos(min_theta)))
    q = beta.cdf(rho**2/(4*r**2), d_sphere/2, d_sphere/2)
    q_min = beta.cdf(rho_min**2/(4*r**2), d_sphere/2, d_sphere/2)
    t = np.random.uniform(size=n, low=q_min, high=q)
    z = r*(2*beta.ppf(t, d_sphere/2, d_sphere/2) - 1)

    #sample the remaining d-1 coordinates
    r_sphere = np.sqrt(r**2 - z**2)
    x_remain = sample_sphere_(n, d_sphere, r=r_sphere)
    x = np.concatenate([z[:, None], x_remain], axis=1)

    if gaussianize:
        y = np.random.normal(size=(n,d))
        y_norm = np.sqrt(np.sum(y**2, axis=1))
        x = x * y_norm[:, None]

    return torch.tensor(x, device=device, dtype=torch.float32)

def gen_linreg_data(seed,batch_size=64,dim=10,n_samples=50,mean=0,std=1, 
                    ws=None, device='cuda',noise_std=None,
                    sequence_transform=None, dim2=-1,norm=False,max_angle=None,min_angle=None):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    if max_angle is None:
        xs = torch.randn(batch_size, n_samples, dim, generator=gen, device=device)
    else:
        xs = sample_cone(batch_size*n_samples, dim, max_angle,min_theta=min_angle)
    xs = xs.reshape(batch_size, n_samples, dim)

    if ws is None:
        ws = mean + std*torch.randn(batch_size, dim, generator=gen, device=device)

    if norm:
        ws = normalize(ws, dim=1)

    ys = torch.einsum('bsd,bd->bs',xs,ws).unsqueeze(-1)
    if noise_std is not None:
        noise = torch.randn(batch_size, n_samples, 1, generator=gen, device=device)
        ys += noise_std*noise

    if sequence_transform is not None:
        assert dim2 > 0
        xs2 = torch.empty(batch_size, n_samples, dim2, device=device)
        for b in range(batch_size):
            xs2[b] = sequence_transform(b, xs[b], dim2=dim2)
        xs = xs2
    concat_dim = dim if dim2 < 0 else dim2
    ys = torch.concat((ys, torch.zeros(batch_size,n_samples,concat_dim-1,device=device)),dim=-1)


    return xs, ys, ws

def gen_twolayer_data(seed,batch_size=64,dim=3,n_samples=50,mean=0,std=1, 
                    ws1=None, ws2=None, device='cuda',noise_std=None,
                    sequence_transform=None, dim2=-1,norm=False):

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    xs = torch.randn(batch_size, n_samples, dim, generator=gen, device=device)

    if ws1 is None:
        ws1 = mean + std*torch.randn(batch_size, dim, generator=gen, device=device)

    if ws2 is None:
        ws2 = mean + std*torch.randn(batch_size, dim, dim, generator=gen, device=device)

    if norm:
        ws = torch.concatenate(ws1, ws2.flatten())
        ws = normalize(ws)
        ws1 = ws[:dim]
        ws2 = ws[dim+1:]
        ws2 = ws.reshape(dim,dim)

    preact = torch.einsum('bsd,bdd->bsd',xs,ws1)
    act = torch.nn.relu(preact)
    ys = torch.einsum('bsd,bd->bs',xs,ws2).unsqueeze(-1)

    if noise_std is not None:
        noise = torch.randn(batch_size, n_samples, 1, generator=gen, device=device)
        ys += noise_std*noise

    if sequence_transform is not None:
        assert dim2 > 0
        xs2 = torch.empty(batch_size, n_samples, dim2, device=device)
        for b in range(batch_size):
            xs2[b] = sequence_transform(b, xs[b], dim2=dim2)
        xs = xs2

    concat_dim = dim if dim2 < 0 else dim2
    ys = torch.concat((ys, torch.zeros(batch_size,n_samples,concat_dim-1,device=device)),dim=-1)

    return xs, ys, ws

def gen_logreg_data(seed,batch_size=64,dim=10,n_samples=50,mean=0,std=1, 
                    ws=None, device='cuda',noise_std=None,
                    sequence_transform=None, dim2=-1,norm=False):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    xs = torch.randn(batch_size, n_samples, dim, generator=gen, device=device)

    if ws is None:
        ws = mean + std*torch.randn(batch_size, dim, generator=gen, device=device)

    if norm:
        ws = normalize(ws, dim=1)

    ys = torch.einsum('bsd,bd->bs',xs,ws).unsqueeze(-1)

    if noise_std is not None:
        noise = torch.randn(batch_size, n_samples, 1, generator=gen, device=device)
        ys += noise_std*noise

    ys = torch.sigmoid(ys)
    ys = (ys >= 0.5).float() #round classes to 0 or 1

    if sequence_transform is not None:
        assert dim2 > 0
        xs2 = torch.empty(batch_size, n_samples, dim2, device=device)
        for b in range(batch_size):
            xs2[b] = sequence_transform(b, xs[b], dim2=dim2)
        xs = xs2
    concat_dim = dim if dim2 < 0 else dim2
    ys = torch.concat((ys, torch.zeros(batch_size,n_samples,concat_dim-1,device=device)),dim=-1)

    return xs, ys, ws