# -*-coding:utf-8-*-
import torch
import numpy as np
import scipy.io as sio
import sys
import os
sys.path.append("Ops/FWI")
from FWI_ops_3d import *
import matplotlib.pyplot as plt
import fwi_utils as ft
import argparse
from scipy import optimize
from obj_wrapper import PyTorchObjective


# get parameters from command line
parser = argparse.ArgumentParser()

parser.add_argument('--exp_name', type=str, default='output_dir')
parser.add_argument('--generate_data', action='store_true')
parser.add_argument('--nIter', type=int, default=1000)
parser.add_argument('--ngpu', type=int, default=1)
args = vars(parser.parse_args())
generate_data = args['generate_data']
exp_name = args['exp_name']
nIter = args['nIter']
ngpu = args['ngpu']


# ========== parameters ============
model_name = "models/marmousi2-3d_little_model-true.mat"
smooth_model_name = "models/marmousi2-3d_little_model-smooth.mat"
model = sio.loadmat(model_name)
smooth_model = sio.loadmat(smooth_model_name)


dz = float(np.squeeze(model["dz"])) # original scale
dx = float(np.squeeze(model["dx"])) # original scale
dy = float(np.squeeze(model["dy"]))
nz = np.squeeze(model["nz"]) # original scale
nx = np.squeeze(model["nx"]) # original scale
ny = np.squeeze(model["ny"])

dt = 0.0025
nSteps = 1400
nPml = 20

nz_pad = int(nz + 2*nPml)
nx_pad = int(nx + 2*nPml)
ny_pad = int(ny + 2*nPml)

Mask = np.zeros((nz_pad, nx_pad, ny_pad))
Mask[nPml:nPml+nz, nPml:nPml+nx, nPml:nPml+ny] = 1.0
th_mask = torch.tensor(Mask, dtype=torch.float32)

f0_vec = [4.5]


ind_src_x = np.arange(4, nx, 8).astype(int)
ind_src_y = np.arange(4, ny, 8).astype(int)
ind_src_z = 2*np.ones(ind_src_x.shape[0]).astype(int) 

n_src_x = len(ind_src_x)
n_src_y = len(ind_src_y)
n_src = n_src_x * n_src_y

ind_src_x = np.repeat(np.expand_dims(ind_src_x, axis=1), n_src_y, axis=1)
ind_src_z = np.repeat(np.expand_dims(ind_src_z, axis=1), n_src_y, axis=1)
ind_src_y = np.repeat(np.expand_dims(ind_src_y, axis=0), n_src_x, axis=0)

# ind_src_x = np.array([[4]])
# ind_src_z = np.array([[2]])
# ind_src_y = np.array([[4]])
# n_src = 1

ind_rec_x = np.arange(3, nx-2, 1).astype(int)
ind_rec_y = np.arange(3, ny-2, 1).astype(int)
ind_rec_z = 10*np.ones(ind_rec_x.shape[0]).astype(int)

n_rec_x = len(ind_rec_x)
n_rec_y = len(ind_rec_y)
n_rec = n_rec_x * n_rec_y

ind_rec_x = np.repeat(np.expand_dims(ind_rec_x, axis=0), n_rec_y, axis=0)
ind_rec_x = np.reshape(ind_rec_x, (1, -1))[0]
ind_rec_z = np.repeat(np.expand_dims(ind_rec_z, axis=1), n_rec_y, axis=1)
ind_rec_z = np.reshape(ind_rec_z, (1, -1))[0]
ind_rec_y = np.repeat(np.expand_dims(ind_rec_y, axis=1), n_rec_x, axis=1)
ind_rec_y = np.reshape(ind_rec_y, (1, -1))[0]


para_fname = './' + exp_name + '/para_file.json'
survey_fname = './' + exp_name + '/survey_file.json'
data_dir_name = './' + exp_name + '/Data'

ft.paraGen_3d(nz_pad, nx_pad, ny_pad, dz, dx, dy, nSteps, dt, f0_vec[0], nPml, para_fname, survey_fname, data_dir_name)  #  保存参数
ft.surveyGen_3d(ind_src_z, ind_src_x, ind_src_y, ind_rec_z, ind_rec_x, ind_rec_y, survey_fname) # 保存探勘参数


Stf = sio.loadmat("models/sourceF_4p5_2_high.mat", squeeze_me=True, struct_as_record=False)["sourceF"]
# print(Stf)
# raise RecursionError("E")
th_Stf = torch.tensor(Stf[:nSteps], dtype=torch.float32, \
  requires_grad=False).repeat(n_src, 1)
Shot_ids = torch.tensor(np.arange(0, n_src), dtype=torch.int32)

opt = {}
opt['nz'] = nz
opt['nx'] = nx
opt['ny'] = ny
opt['nPml'] = nPml
opt['para_fname'] = para_fname

########################## 3D_Forward #############################

if generate_data == True:
  vp_true = np.ascontiguousarray(np.reshape(model['vp'], (nz, nx, ny), order='F'))
  # cs_true = np.ascontiguousarray(np.reshape(model['vp'], (nz, nx, ny), order='F')) * 0.45
  vs_true = np.zeros((nz, nx, ny))
  # print(f'cp_true shape = {cp_true.shape}')
  # plt.imshow(cp_true, cmap='RdBu_r')
  # plt.colorbar()
  # plt.savefig('vp.png')
  # rho_true_pad = np.ascontiguousarray(np.reshape(model['rho'], (nz, nx, ny), order='F'))
  rho_true = 2.500 * np.ones((nz, nx, ny))

  th_vp = torch.tensor(vp_true, dtype=torch.float32, requires_grad=False)
  th_vs = torch.tensor(vs_true, dtype=torch.float32, requires_grad=False)
  th_rho = torch.tensor(rho_true, dtype=torch.float32, requires_grad=False)

  fwi_obscalc = FWI_obscalc(th_vp, th_vs, th_rho, th_Stf, opt)
  fwi_obscalc(Shot_ids, ngpu=ngpu)
  sys.exit("end of generate data")


########################## 3D_Inversion#############################

vp_init = np.ascontiguousarray(np.reshape(smooth_model['vp'], (nz, nx, ny), order='F'))

# vs_init = np.ascontiguousarray(np.reshape(smooth_model['vp'], (nz, nx, ny), order='F')) * 0.45
vs_init = np.ascontiguousarray(np.zeros((nz, nx, ny)))
# rho_init = np.ascontiguousarray(np.reshape(smooth_model['rho'], (nz, nx, ny), order='F'))
rho_init = 2.500 * np.ones((nz, nx, ny))

rho_init = rho_init * 1e3

th_vp_inv = torch.tensor(vp_init, dtype=torch.float32, requires_grad=True)
th_vs_inv = torch.tensor(vs_init, dtype=torch.float32, requires_grad=False)
th_rho_inv = torch.tensor(rho_init, dtype=torch.float32, requires_grad=False)


fwi = FWI_3d(th_vp_inv, th_vs_inv, th_rho_inv, th_Stf, opt, Mask=th_mask)

compLoss = lambda : fwi(Shot_ids, ngpu=ngpu)

obj = PyTorchObjective(fwi, compLoss)


__iter = 0
result_dir_name = './' + exp_name + '/Results_BFGS'
figure_dir_name = './' + exp_name + '/Figures_BFGS'

def save_prog(x):
    global __iter
    if __iter % 2 == 0:
        os.makedirs(result_dir_name, exist_ok=True)
        os.makedirs(figure_dir_name, exist_ok=True)

        plt.imshow(fwi.Vp.cpu().detach().numpy()[:, 30, :])
        plt.title("vp")
        plt.savefig(figure_dir_name + "/vp" + str(__iter) + '.png')
        plt.imshow((fwi.Vs.cpu().detach().numpy())[:, 30, :])
        plt.title("vs")
        plt.savefig(figure_dir_name + '/Vs' + str(__iter) + '.png')
        plt.imshow(fwi.Rho.cpu().detach().numpy()[:, 30, :])
        plt.title("Rho")
        plt.savefig(figure_dir_name + '/Rho' + str(__iter) + '.png')

        with open(result_dir_name + '/lossRsp.txt', 'a') as text_file:
            text_file.write("%d %s\n" % (__iter, obj.f))
            sio.savemat(result_dir_name + '/result' + str(__iter) + '.mat', \
            {'cp':fwi.Vp.cpu().detach().numpy(), 'cs':fwi.Vs.cpu().detach().numpy(), 'Rho':fwi.Rho.cpu().detach().numpy()})
            # sio.savemat(result_dir_name + '/grad' + str(__iter) + \
            # '.mat', {'grad_cp':fwi.Vp.grad.cpu().detach().numpy(), 'grad_cs':fwi.Vs.grad.cpu().detach().numpy(), 'grad_Rho':fwi.Rho.grad.cpu().detach().numpy()})
    __iter = __iter + 1

maxiter = nIter  # 100
optimize.minimize(obj.fun, obj.x0, method='L-BFGS-B', jac=obj.jac, bounds=obj.bounds, \
  tol=None, callback=save_prog, options={'disp': True, 'iprint': 101, \
  'gtol': 1e-012, 'maxiter': maxiter, 'ftol': 1e-16, 'maxcor': 30, 'maxfun': 15000})

