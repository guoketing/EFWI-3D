from multiprocessing.pool import RUN
import torch
import torch.nn as nn
import numpy as np 
from torch.utils.cpp_extension import load
import matplotlib.pyplot as plt
import os
from scipy import optimize
import fwi_utils as ft
from collections import OrderedDict

abs_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(abs_path, 'Src_3d')
os.makedirs(path+'/build/', exist_ok=True)
# import fwi

def load_fwi(path):
    fwi = load(name="fwi", sources=[path+'/Torch_Fwi.cpp', path+'/Parameter.cpp', path+'/libCUFD.cu', path+'/stress.cu',
                                    path+'/velocity.cu', path+'/stress_adj.cu', path+'/velocity_adj.cu',
                                    path+'/Model.cu', path+'/Cpml.cu', path+'/utilities.cu', path+'/Src_Rec.cu',
                                    path+'/Boundary.cu'],
            extra_cflags=[
                '-O3 -fopenmp -lpthread'
            ],
            extra_include_paths=['/usr/local/cuda/include', path+'/rapidjson'],
            extra_ldflags=['-L/usr/local/cuda/lib64 -lnvrtc -lcuda -lcudart -lcufft'],
            build_directory=path+'/build/',
            verbose=True)
    return fwi

fwi_ops = load_fwi(path)


class FWIFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Lambda, Mu, Rho, Stf, ngpu, Shot_ids, para_fname):
        outputs = fwi_ops.backward(Lambda, Mu, Rho, Stf, ngpu, Shot_ids, para_fname)
        ctx.outputs = outputs[1:]
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_misfit):
        grad_Lambda, grad_Mu, grad_Rho, grad_stf = ctx.outputs
        return grad_Lambda, grad_Mu, grad_Rho, grad_stf, None, None, None


class FWI_3d(torch.nn.Module):
    def __init__(self, Vp, Vs, Rho, Stf, opt, Mask=None):
        super(FWI_3d, self).__init__()

        self.nz = opt['nz'] # 134
        self.nx = opt['nx'] # 384
        self.ny = opt['ny'] # 384
        self.nPml = opt['nPml'] # 32

        self.Bounds = {}
        Vp_pad, Vs_pad, Rho_pad = ft.padding_3d(Vp, Vs, Rho, self.nz, self.nx, self.ny, self.nPml)
        Vp_ref = Vp_pad.clone().detach()
        Vs_ref = Vs_pad.clone().detach()
        Rho_ref = Rho_pad.clone().detach()
        self.register_buffer('Vp_ref', Vp_ref)
        self.register_buffer('Vs_ref', Vs_ref)
        self.register_buffer('Rho_ref', Rho_ref)
        if Vp.requires_grad:
            self.Vp = nn.Parameter(Vp)
        else:
            self.Vp = Vp
        if Vs.requires_grad:
            self.Vs = nn.Parameter(Vs)
        else:
            self.Vs = Vs
        if Rho.requires_grad:
            self.Rho = nn.Parameter(Rho)
        else:
            self.Rho = Rho

        if Mask == None:
            self.Mask = torch.ones((self.nz+2*self.nPml, \
                self.nx+2*self.nPml), dtype=torch.float32)
        else:
            self.Mask = Mask

        self.Stf = Stf
        self.para_fname = opt['para_fname']

    def forward(self, Shot_ids, ngpu=1):
        Vp_pad, Vs_pad, Rho_pad = ft.padding_3d(self.Vp, self.Vs, self.Rho, self.nz, self.nx, self.ny, self.nPml)
        Vp_mask_pad = self.Mask * Vp_pad + (1.0 - self.Mask) * self.Vp_ref
        Vs_mask_pad = self.Mask * Vs_pad + (1.0 - self.Mask) * self.Vs_ref
        Rho_mask_pad = self.Mask * Rho_pad + (1.0 - self.Mask) * self.Rho_ref
        Rho = Rho_mask_pad / 1e3
        Lambda = (Vp_mask_pad**2 - 2.0 * Vs_mask_pad**2) * Rho
        Mu = Vs_mask_pad**2 * Rho
        return FWIFunction.apply(Lambda, Mu, Rho, self.Stf, ngpu, Shot_ids, self.para_fname)

class FWI_obscalc(torch.nn.Module):
    def __init__(self, Vp, Vs, Rho, Stf, opt):
        super(FWI_obscalc, self).__init__()
        Vp_pad, Vs_pad, Rho_pad = ft.padding_3d(Vp, Vs, Rho, opt['nz'], opt['nx'], opt['ny'], opt['nPml'])
        self.Lambda = (Vp_pad**2 - 2.0 * Vs_pad**2) * Rho_pad
        self.Mu = Vs_pad**2 * Rho_pad
        self.Rho = Rho_pad
        self.Stf = Stf
        self.para_fname = opt['para_fname']

    def forward(self, Shot_ids, ngpu=1):

        fwi_ops.obscalc(self.Lambda, self.Mu, self.Rho, self.Stf, ngpu, Shot_ids, self.para_fname)
