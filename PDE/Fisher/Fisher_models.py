##this file defines all utility functions to use to train a multihead
##neural network to solve a series of PDEs of form ut - D*uxx = f
##and later use perturbation method and one-shot TL to solve the Fisher's Equation 

#import all required packages
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurodiffeq import diff
from neurodiffeq.neurodiffeq import unsafe_diff as unsafe_diff
from neurodiffeq.conditions import IVP, DirichletBVP, DirichletBVP2D, BundleIVP, BundleDirichletBVP
from neurodiffeq.solvers import Solver1D, Solver2D, BundleSolver1D
from neurodiffeq.networks import FCNN, SinActv
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.generators import Generator1D, Generator2D

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import time
from tqdm.auto import tqdm

##define the grids where the function u is evaluated
N = 60 #N is the number of grids in each dimension, in total there will be (N+1)^2 grid points
x_grid = np.arange(0, 1 + .5/N, 1/N)
t_grid = np.arange(0, 1 + .5/N, 1/N)
X_grid = []
for t in t_grid:
  for x in x_grid:
    X_grid.append([x, t])

X_grid = torch.Tensor(np.array(X_grid))

#original shape 2-128-128-256-256-512-(2, 256) - k heads
class Multihead_model(nn.Module):
  def __init__(self, k, act = nn.Tanh(), bias=False): #silu
    super().__init__()
    self.act = act
    self.linear1 = nn.Linear(2, 128)
    self.linear2 = nn.Linear(128, 128)
    self.linear3 = nn.Linear(128, 256)
    self.linear4 = nn.Linear(256, 256)
    self.linear5 = nn.Linear(256, 512)
    ##define k final layers without bias
    self.final_layers = nn.ModuleList(
        [nn.Linear(256, 1, bias=bias) for _ in range(k)]
    )
    self.k = k

  #it returns the output of the network and the hidden state
  def forward(self, x):
    fc1 = self.act(self.linear1(x))
    out = self.act(self.linear2(fc1))
    out = self.act(self.linear3(out))
    #out = out+fc1 #skip connection
    out = self.act(self.linear4(out))
    out = self.act(self.linear5(out)) #out of shape (N, 512)
    out1 = out[:, :256] #shape (N, 256)
    out2 = out[:, 256:] #shape (N, 256)
    output = []
    for i in range(self.k):
      first = self.final_layers[i](out1) #shape (N, 1)
      second = self.final_layers[i](out2) #shape (N, 1)
      concat = torch.cat((first, second), axis=1) #shape (N, 2)
      output.append(concat)
    return torch.stack(output), out #of shape (k, N, 2) and (N, 512) repectively
    
    
##define the forcing functions, should be a list of forcing functions
##the returned value has the same dimension as the input x and t

##define the forcing functions, should be a list of forcing functions
##the returned value has the same dimension as the input x and t

##returns function of form f(x, t) = 2At
def forcing_decorator(A):
  def force(x, t):
    return 2*A*t
  return force

#returns function of form f(x, t) = Atx(x-1) + b
def function_decorator(A, b = 0):
  def func(x, t):
    return A*t*x*(x-1) + b
  return func

#returns function of form f(x, t) = 2A(x^2 + t^2 - cx -x)
def forcing_decorator2(A, c):
  def force(x, t):
    return 2*A*(x**2 + t**2 - c*t - x)
  return force

#returns function of form f(x, t) = Atx(x-1)(t-c) + b
def function_decorator2(A, b, c):
  def func(x, t):
    return A*t*x*(x-1)*(t-c) + b
  return func

def forcing_decorator_trig(A, k1, k2, D):
  def force(x, t):
    Sink1 = torch.sin(k1*torch.pi*x)
    Sink2 = torch.sin(k2*torch.pi*t)
    Cosk2 = torch.cos(k2*torch.pi*t)
    return A*k2*torch.pi*Sink1*Cosk2 + A*D*(torch.pi**2)*(k1**2)*Sink1*Sink2
  return force

def truth_decorator_trig(a, k1, k2, b=0):
  def force(x, t):
    return a*torch.sin(torch.pi*k1*x)*torch.sin(torch.pi*k2*t) + b
  return force


def loss(model, interior_grid, x_boundary_num, t_boundary_num, boundary_value,
         Forcing_functions, truth_functions, D=1, pde_weight=1, bc_weight=1, data_weight=1, method='chebyshev'):
    #use 2d chebyshev generator to generate sample points in the interior
    generator= Generator2D(grid=interior_grid, method=method)
    samples = generator.get_examples()
    #convert this sample points into input to the network and requires gradients
    x = samples[0].unsqueeze(1) #(N, 1)
    t = samples[1].unsqueeze(1) #(N, 1)
    x[x<0] = 0; x[x>1] = 1
    x.requires_grad_()
    t.requires_grad_()
    input_tensor = torch.cat([x, t], dim=1)

    ##evaluate the network on these points
    output_tensor, _ = model(input_tensor) #shape (k, N, 2)

    #separate the u, y, z parts of the output of the network
    u = output_tensor[:,:,0].T; #shape (N, k)
    y = output_tensor[:,:,1].T; #shape (N, k)

    ##compute the gradients
    dudt = torch.cat([diff(u[:,i].reshape(-1, 1), t) for i in range(u.shape[1])], dim=1) #shape (N, k)
    dudx = torch.cat([diff(u[:,i].reshape(-1, 1), x) for i in range(u.shape[1])], dim=1) #shape (N, k)
    dydx = torch.cat([diff(y[:,i].reshape(-1, 1), x) for i in range(y.shape[1])], dim=1) #shape (N, k)
    
    ##compute the forcing function on N data points across k heads
    force = torch.cat([Forcing_functions[i](x, t) for i in range(len(Forcing_functions))], dim=1) #(N, k)

    ##compute the pde residual
    residual = torch.cat(
        [
            (dudt - D*dydx - force).unsqueeze(2),
            (dudx - y).unsqueeze(2),
        ],
        dim=2
    ) #shape (N, k, 3)
    ##compute the pde loss
    pde_loss = F.mse_loss(residual, torch.zeros_like(residual))

    ##forward pass for the BC condition
    ##sample points from the boundary first
    x_samples = Generator1D(size=x_boundary_num, method='chebyshev').get_examples()
    t_samples = Generator1D(size=t_boundary_num, method='chebyshev').get_examples()
    x_boundary = torch.cat([
        torch.zeros_like(t_samples),
        torch.ones_like(t_samples),
        x_samples,
    ]).unsqueeze(1)
    t_boundary = torch.cat([
        t_samples,
        t_samples,
        torch.zeros_like(x_samples),
    ]).unsqueeze(1)
    x_boundary.requires_grad_()
    t_boundary.requires_grad_()
    input_boundary = torch.cat([x_boundary, t_boundary], dim=1) #(N_boundary, 2)
    ##evaluate the neural network at the boundary
    output_boundary, _ = model(input_boundary) #(k, N_boundary, 2)
    u0_boundary = output_boundary[:,:,0] #(k, N_boundary)
    
    if isinstance(boundary_value, (int, float)):
      truth_boundary = torch.ones_like(u0_boundary)*boundary_value
    elif isinstance(boundary_value[0], (int, float)):
      truth_boundary = torch.ones_like(u0_boundary) * torch.tensor(np.array(boundary_value)[:, np.newaxis])
      truth_boundary.to(u0_boundary.device)
    #if the boundary_value is a list of functions
    else: 
      truth_boundary = torch.stack([truth(input_boundary[:,0], input_boundary[:, 1]) for truth in boundary_value])
    bc_loss = F.mse_loss(u0_boundary, truth_boundary)

    ##compute the data loss
    truth = torch.cat([truth_functions[i](x, t) for i in range(len(truth_functions))], dim=1) #(N, k)
    data_loss = F.mse_loss(u, truth)

    #sum the weighted loss
    total_loss = pde_weight*pde_loss + bc_weight*bc_loss + data_weight*data_loss
    return total_loss, pde_loss, bc_loss, data_loss

def train(model, optimizer, lossfn, num_iter, Forcing_functions, truth_functions,
          boundary_value, D = 1, interior_grid=(30, 30), x_boundary_num=100, t_boundary_num=100,
          every=100, pde_weight=1, bc_weight=1, data_weight=1, scheduler=None, method='chebyshev'):
  loss_trace = []; pde_loss_trace = []; bc_loss_trace = []; data_loss_trace = []
  for i in tqdm(range(num_iter)):
    optimizer.zero_grad()
    #evaluate the loss
    total, pde, bc, data = loss(model, interior_grid, x_boundary_num, t_boundary_num, boundary_value,
                Forcing_functions, truth_functions, D = D,
                pde_weight=pde_weight, bc_weight=bc_weight, data_weight=data_weight, method=method)
    #take the step
    total.backward()
    optimizer.step()
    if scheduler is not None:
      scheduler.step()
    #record
    loss_trace.append(total.item())
    pde_loss_trace.append(pde.item())
    bc_loss_trace.append(bc.item())
    data_loss_trace.append(data.item())
    #print
    if (i+1)%every == 0:
      print("{}th Iter: total {}, pde {}, bc {}, data {}".format(i+1, total.item(),
                                                                 pde.item(), bc.item(), data.item()))
  return loss_trace, pde_loss_trace, bc_loss_trace, data_loss_trace

def plot_solutions12(solutions, title, savfig = None, figsize=(12, 3.5), subtitle_list = None):
  ##plot the 12 solutions
  fig, ax = plt.subplots(2, 6, figsize=figsize);

  for i in range(12):
    global_min = solutions[i].min()
    global_max = solutions[i].max()
    j = i // 6
    k = i % 6
    # Create color map
    cmap = plt.get_cmap('viridis');  # You can choose any colormap you prefer
    # Create colorbar
    im = ax[j][k].imshow(solutions[i][::-1, :], cmap=cmap, vmin=global_min, vmax=global_max);
    ax[j][k].axis('off')
    cbar = fig.colorbar(im, ax=ax[j][k], shrink=0.9, aspect=8)
    cbar.ax.tick_params(labelsize=6)
    if subtitle_list:
      ax[j][k].set_title(subtitle_list[i],
                         fontdict={'family': 'serif', 'color':  'darkred', 'weight': 'bold', 'size': 10})

  plt.suptitle(title);
  plt.subplots_adjust(top=0.93)
  if savfig is not None:
    plt.savefig(savfig, format='eps')
    
def plot_solutions16(solutions, title, savfig = None, figsize=(16, 3.5), subtitle_list = None):
  ##plot the 16 solutions
  fig, ax = plt.subplots(2, 8, figsize=figsize);

  for i in range(16):
    global_min = solutions[i].min()
    global_max = solutions[i].max()
    j = i // 8
    k = i % 8
    # Create color map
    cmap = plt.get_cmap('viridis');  # You can choose any colormap you prefer
    # Create colorbar
    im = ax[j][k].imshow(solutions[i][::-1, :], cmap=cmap, vmin=global_min, vmax=global_max);
    ax[j][k].axis('off')
    cbar = fig.colorbar(im, ax=ax[j][k], shrink=0.9, aspect=8)
    cbar.ax.tick_params(labelsize=6)
    if subtitle_list:
      ax[j][k].set_title(subtitle_list[i],
                         fontdict={'family': 'serif', 'color':  'darkred', 'weight': 'bold', 'size': 10})

  plt.suptitle(title);
  plt.subplots_adjust(top=0.93)
  if savfig is not None:
    plt.savefig(savfig, format='eps')

def plot_solutions24(solutions, title, savfig = None, figsize=(12, 7), subtitle_list = None):
  ##plot the 24 solutions
  fig, ax = plt.subplots(4, 6, figsize=figsize);

  for i in range(24):
    global_min = solutions[i].min()
    global_max = solutions[i].max()
    j = i // 6
    k = i % 6
    # Create color map
    cmap = plt.get_cmap('viridis');  # You can choose any colormap you prefer
    # Create colorbar
    im = ax[j][k].imshow(solutions[i][::-1, :], cmap=cmap, vmin=global_min, vmax=global_max);
    ax[j][k].axis('off')
    cbar = fig.colorbar(im, ax=ax[j][k], shrink=0.9, aspect=8)
    cbar.ax.tick_params(labelsize=6)
    if subtitle_list:
      ax[j][k].set_title(subtitle_list[i],
                         fontdict={'family': 'serif', 'color':  'darkred', 'weight': 'bold', 'size': 10})

  plt.suptitle(title);
  plt.subplots_adjust(top=0.93)
  if savfig is not None:
    plt.savefig(savfig, format='eps')

##this function plots the loss trace
##it takes four arrays of losses: total loss, pde loss, BC loss and data loss (optional)
def plot_loss(loss_trace, pde_trace, bc_trace, data_trace=None, figsize=(15, 5), path=None):
    num_iter = len(loss_trace)
    _, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].plot(range(1, num_iter+1), np.log10(loss_trace), label='Total Loss');
    ax[1].plot(range(1, num_iter+1), np.log10(pde_trace), label='pde Loss');
    ax[1].plot(range(1, num_iter+1), np.log10(bc_trace), label="BC Loss");
    if data_trace is not None:
        ax[1].plot(range(1, num_iter+1), np.log10(data_trace), label='data Loss');

    ax[0].set_xlabel("Number of iterations");
    ax[0].set_ylabel("Log loss");
    ax[0].set_title("Log Total Loss Value vs. Iteration");
    ax[0].grid();
    ax[0].legend();
    ax[1].set_xlabel("Number of iterations");
    ax[1].set_ylabel("Log loss");
    if data_trace is not None:
        ax[1].set_title("Log pde,data and BC Loss Value vs. Iteration");
    else:
        ax[1].set_title("Log pde, BC Loss Value vs. Iteration");
    ax[1].grid();
    ax[1].legend();
    if path is not None:
        plt.savefig(path, format='eps');

##compute the relative error between the NN solution and the truth value
##it is defined as the ratio between the mean of absolte error and the mean of absolute value of truth
def relative_error(NN_solution, truth_solution):
  return abs(np.array(NN_solution) - np.array(truth_solution)).mean() / abs(np.array(truth_solution)).mean()

##this function generate interior tensors to evaluate the Hs
##I is the number of samples in each dimension
def generate_interior_tensor(I=60, require_grad = True):
  N_i = I**2
  generator= Generator2D(grid=(I, I), method='equally-spaced',
                        xy_min=(1/200, 1/200), xy_max = (1-1/200, 1-1/200))
  samples = generator.get_examples()
  #convert this sample points into input to the network and requires gradients
  x = samples[0].unsqueeze(1) #(N, 1)
  t = samples[1].unsqueeze(1) #(N, 1)
  x = x.cpu(); t = t.cpu()
  if require_grad:
    x.requires_grad_()
    t.requires_grad_()
  interior_tensor = torch.cat([x, t], dim=1)
  return (x, t, interior_tensor)

##this function generate boundary points to evaluate the Hs
#B is the number of samples in each boundary, 3B samples in total
def generate_boundary_tensor(B=200, require_grad=True, method='uniform'):
  N_b = 3*B
  x_samples = Generator1D(size=B, method=method).get_examples()
  t_samples = Generator1D(size=B, method=method).get_examples()
  x_boundary = torch.cat([
          torch.zeros_like(t_samples),
          torch.ones_like(t_samples),
          x_samples,
  ]).unsqueeze(1)
  t_boundary = torch.cat([
          t_samples,
          t_samples,
          torch.zeros_like(x_samples),
  ]).unsqueeze(1)
  x_boundary = x_boundary.cpu(); t_boundary = t_boundary.cpu()
  if require_grad:
    x_boundary.requires_grad_()
    t_boundary.requires_grad_()
  boundary_tensor = torch.cat([x_boundary, t_boundary], dim=1) #(N_boundary, 2)
  return (x_boundary, t_boundary, boundary_tensor)

##function to compute the gradient of H w.r.t. x
def compute_Hx(H, x):
  output = []
  for i in range(H.shape[1]):
    output.append(diff(H[:,i].reshape(-1, 1), x).detach().numpy())
  return np.concatenate(output, axis=1)

##function to compute the gradient of H w.r.t. t
def compute_Ht(H, t):
  output = []
  for i in range(H.shape[1]):
    output.append(diff(H[:,i].reshape(-1, 1), t).detach().numpy())
  return np.concatenate(output, axis=1)

def compute_AH(A, H):
  N, W_size = H.shape
  A_reshaped = A.reshape(1, 2, 2)
  # Reshape H to (3600, 2, 256) to match the dimensions of A
  H_reshaped = H.reshape(-1, 2, W_size)
  # Perform the multiplication
  AH = np.matmul(A_reshaped, H_reshaped)
  # Reshape the result back
  AH = AH.reshape(-1, W_size)
  return AH

##this function computes all the H matrices and returns them as a dictionary
##H_dict = {H, H_b, Hx, Ht, Hx_b, Ht_b, I, B}
##these matrices are evaluated on predefined meshgrids
##I is the number of interior sample points in each dimension
##B is the number of samples on each of the three boundaries
##bias T or F, whether the model has bias in the final layers
def compute_H_dict(model, I, B, bias, D = 1):
  ##change the model to cpu
  model.to('cpu')
  ##generate samples
  ##define the interior set used to do transfer learning on cpu
  x, t, interior_tensor = generate_interior_tensor(I=I, require_grad = True)
  ##define the boundary set used to do transfer learing on cpu
  x_boundary, t_boundary, boundary_tensor = generate_boundary_tensor(B=B, require_grad=True, method='equally-spaced')
  #compute the hidden space H evaluated at the interior
  _, H = model(interior_tensor) # shape (I, 2W)
  #compute the hidden space evaluated at the boundary
  _, H_b = model(boundary_tensor) #shape (B, 2W)
  #compute the gradients of the hidden H in the interior data points
  print("Differentiating H w.r.t. x now...")
  Hx = compute_Hx(H, x) #of shape (I, 2W)
  print("Finished computing Hx.")
  print("Differentiating H w.r.t. t now...")
  Ht = compute_Ht(H, t) #of shape (I, 2W)
  print("Finished computing Ht")
  #compute the gradients of the hidden H in the boudnary
  print("Differentiating H_b w.r.t. x now...")
  Hx_b = compute_Hx(H_b, x_boundary) #shape (B, 2W)
  print("Finished computing Hx_b.")
  print("Differentiating H_b w.r.t. t now...")
  Ht_b = compute_Ht(H_b, t_boundary) #shape (B, 2W)
  print("Finished computing Ht_b")
  ##detach the H
  H = H.detach().numpy() #shape (I, 2W)
  H_b = H_b.detach().numpy() #shape (B, 2W)
  H = H.reshape(2*H.shape[0], -1) #transform all H into shape (2I, W)
  H_b = H_b.reshape(2*H_b.shape[0], -1)
  Hx = Hx.reshape(2*Hx.shape[0], -1) #shape (2I, W)
  Ht = Ht.reshape(2*Ht.shape[0], -1) #shape (2I, W)
  Hx_b = Hx_b.reshape(2*Hx_b.shape[0], -1) #shape (2B, W)
  Ht_b = Ht_b.reshape(2*Ht_b.shape[0], -1) #shape (2B, w)
  #now add another dimension to the H and Hx Ht if needed
  #all zeros for Hx and Ht and all ones for H
  if bias:
    H = np.hstack((H, np.ones((H.shape[0], 1))))
    H_b = np.hstack((H_b, np.ones((H_b.shape[0], 1))))
    Hx = np.hstack((Hx, np.zeros((Hx.shape[0], 1))))
    Ht = np.hstack((Ht, np.zeros((Ht.shape[0], 1))))
    Hx_b = np.hstack((Hx_b, np.zeros((Hx_b.shape[0], 1))))
    Ht_b = np.hstack((Ht_b, np.zeros((Ht_b.shape[0], 1))))
  ##define the matrices A
  A1 = np.array([[0, -D], [1, 0]]);
  A2 = np.array([[1, 0], [0, 0]]);
  A3 = np.array([[0, 0], [0, 1]]);
  A1Hx = compute_AH(A1, Hx) #of shape (2I, W)
  A2Ht = compute_AH(A2, Ht) #shape (2I, W)
  A3H = compute_AH(A3, H) #shape (2I, W)
  H_star = A1Hx + A2Ht - A3H #(2I, W)
  H_dict =  {'H': H, 'H_b': H_b, 'Hx': Hx, 'Ht': Ht, 'Hx_b': Hx_b, 'Ht_b': Ht_b, 'I': I, 'B': B,
             'A1Hx': A1Hx, 'A2Ht': A2Ht, 'A3H': A3H, 'H_star': H_star}
  return H_dict

##this function compute and inverts the matrix M
def compute_M(H_dict):
  I = H_dict['I']; B = H_dict['B']
  ##obtain the first rows of H_b
  H_b_0 = H_dict['H_b'][[2*i for i in range(3*B)]]
  N_i = I**2; N_b = 3*B
  M = (H_dict['H_star'].T@H_dict['H_star'])/N_i + (H_b_0.T@H_b_0)/N_b # shape (W, W)
  Minv = np.linalg.pinv(M) # shape (W, W)
  return M, Minv

#b0: scalar, constant boundary value for u0
#forcing_f: function handler for the forcing function
##x, t the input samples
def compute_TLW(b0, forcing_f, Minv, H_dict, x, t, x_boundary=None, t_boundary=None):
  ##obtain the first rows of H_b
  H_b_0 = H_dict['H_b'][[2*i for i in range(3*H_dict['B'])]]
  ##obtain the forcing fuunction evaluated at interior points of shape (2I, 1)
  f_values = forcing_f(x, t).detach().numpy() #shape (I, 1)
  F_values = np.concatenate([f_values.T, np.zeros((f_values.T).shape)]).T.reshape(-1, 1) #(2I, 1)
  N_i = H_dict['I']**2; N_b = 3*H_dict['B']
  if isinstance(b0, (int, float)): #if we have constant boundary condition 
    R = (H_dict['H_star'].T@F_values)/N_i +  ((b0*H_b_0.T).sum(axis=1)/N_b).reshape(-1, 1)
  else: #if b0 is a function 
    if x_boundary is None or t_boundary is None:
        raise ValueError('x_boundary and t_boundary cannot be None when the boundary condition is not constant')
    b0 = b0(x_boundary, t_boundary).detach().numpy() 
    R = (H_dict['H_star'].T@F_values)/N_i + (H_b_0.T@b0)/N_b
  #compute W
  W = Minv@R #shape (256, 1)
  return W

#b0: scalar, constant boundary value for u0
#forcing_f: function handler for the forcing function
##x, t the input samples
def compute_TLW_from_fvalues(b0, f_values, Minv, H_dict, x_boundary=None, t_boundary=None):
  ##obtain the first rows of H_b
  H_b_0 = H_dict['H_b'][[2*i for i in range(3*H_dict['B'])]]
  F_values = np.concatenate([f_values.T, np.zeros((f_values.T).shape)]).T.reshape(-1, 1) #(2I, 1)
  N_i = H_dict['I']**2; N_b = 3*H_dict['B']
  if isinstance(b0, (int, float)): #if we have constant boundary condition 
    R = (H_dict['H_star'].T@F_values)/N_i +  ((b0*H_b_0.T).sum(axis=1)/N_b).reshape(-1, 1)
  else: #if b0 is a function 
    if x_boundary is None or t_boundary is None:
        raise ValueError('x_boundary and t_boundary cannot be None when the boundary condition is not constant')
    b0 = b0(x_boundary, t_boundary).detach().numpy() 
    R = (H_dict['H_star'].T@F_values)/N_i + (H_b_0.T@b0)/N_b
  #compute W
  W = Minv@R #shape (256, 1)
  return W

##I is the number of data point in each dimension: total I**2 interrior data point
def compute_solution(H, W, I):
  H_ = H[[2*i for i in range(round(H.shape[0]/2))],:]
  return (H_@W).reshape(I, -1)

##this function compute the after loss
def after_loss(H_dict, W, forcing_f, b0, x, t, x_boundary=None, t_boundary=None):
  H = H_dict['H']; H_b = H_dict['H_b']; H_star = H_dict['H_star']
  N_b = 3*H_dict['B']
  f_values = forcing_f(x, t).detach().numpy() #shape (I, 1)
  F_values = np.concatenate([f_values.T, np.zeros((f_values.T).shape)]).T.reshape(-1, 1) #(2I, 1)
  ##compute the boundary vector b
  if isinstance(b0, (int, float)):
    b_vec = (np.ones((1, N_b))*b0).reshape(-1, 1)
  else:
    if x_boundary is None or t_boundary is None:
        raise ValueError("x_boundary and t_boundary cannot be none")
    b_vec = b0(x_boundary, t_boundary).detach().numpy() 
  boundary_values = (H_b@W)[[2*i for i in range(N_b)]]
  return ((H_star@W - F_values)**2).mean() + ((boundary_values - b_vec)**2).mean()

def after_loss_from_fvalues(H_dict, W, f_values, b0):
  H = H_dict['H']; H_b = H_dict['H_b']; H_star = H_dict['H_star']
  N_b = 3*H_dict['B']
  F_values = np.concatenate([f_values.T, np.zeros((f_values.T).shape)]).T.reshape(-1, 1) #(2I, 1)
  ##compute the boundary vector b
  b_vec = (np.ones((1, N_b))*b0).reshape(-1, 1)
  boundary_values = (H_b@W)[[2*i for i in range(N_b)]]
  return ((H_star@W - F_values)**2).mean() + ((boundary_values - b_vec)**2).mean()

##this function computes the linear combination of a list of weights W
## W = \sum_i=0^p lambda^i W_i
def combine_W(W_list, r):
  W = W_list[0]
  for i in range(1, len(W_list)):
    W += (r**i)*W_list[i]
  return W

##this function compute the boundary value for each function
def compute_each_b(b, r, p):
  deno = 1
  for i in range(1, p+1):
    deno += r**i
  return b/deno

##this function uses perturbation method to solve a non-linear Klein-Gordon Equation
def Fisher_solver(H_dict, Minv, f0_values, b, r, H, p=11, display=False):
  t0 = time.time();
  ##compute the BC condition on each PDE of the series
  b_each = compute_each_b(b, r, p)
  ##container to keep track of everything
  W_list = []; u_list = []; f_list = []

  ##first compute the first u0 and W0
  W0 = compute_TLW_from_fvalues(b_each, f0_values, Minv, H_dict) #shape (257, 1)
  ##compute the u0
  u0 = compute_solution(H_dict['H'], W0, I=H_dict['I'])

  W_list.append(W0); u_list.append(u0); f_list.append(f0_values)

  ##compute the record all the remaining p PDEs
  for i in range(1, p+1):
    result = np.zeros(u_list[0].shape)
    if i%2 == 0:
      for j in range(i//2):
        result -= 2*u_list[j]*u_list[i-1-j]
    else:
      S = i//2
      for j in range(S):
        result -= 2*u_list[j]*u_list[i-1-j]
      result -= u_list[S]**2
    result += u_list[i-1]
    ##reshape result as an output of f_values
    result = result.reshape(-1, 1)
    ##compute the W
    Wi = compute_TLW_from_fvalues(b_each, result, Minv, H_dict) #shape (257, 1)
    ##compute the ui
    ui = compute_solution(H_dict['H'], Wi, I=H_dict['I'])
    #record
    W_list.append(Wi); u_list.append(ui); f_list.append(result)
  t1 = time.time()
  ##compute the final Weight W
  W = combine_W(W_list, r)
  ##compute the solution
  sol = compute_solution(H, W, I=round((H.shape[0]/2)**.5))

  ##put everything together
  result = {
      'W_list': W_list,
      'u_list': u_list,
      'f_list': f_list,
      'W': W,
      'sol': sol
  }
  if display:
    print("Function finisheds in {} seconds. On average, each PDE is solved using {} seconds".format(round(t1-t0, 6), round((t1-t0)/(p+1), 6)))
  return result
  
##this function compute the Fisher's Equation Loss
##returns a  dictionary that contains: 1. total loss 2. PDE loss, 3. BC loss
def Fisher_loss(model, W, r, forcing_function, b0, D = 1, I=60, B=200, bias=True):
  x, t, interior_tensor = generate_interior_tensor(I=I, require_grad = True)
  ##define the boundary set used to do transfer learing on cpu
  x_boundary, t_boundary, boundary_tensor = generate_boundary_tensor(B=B, require_grad=True)

  #compute the values at interior points
  _, H  = model(interior_tensor)
  ##detach the H
  H = H.reshape(2*H.shape[0], -1) #transform all H into shape (2I^2, W)
  H = H[[2*i for i in range(I**2)]] ##only pick the entries that corresponds to u
  if bias: H = torch.hstack((H, torch.ones((H.shape[0], 1))))
  ##compute the u = HW
  u = H@torch.tensor(W) ##(I**2, 1)

  #compute the second order derivatives
  ut = diff(u, t) ##(I**2, 1)
  uxx = diff(diff(u, x), x) ##(I**2, 1)
  
  f_term = forcing_function(x, t)
  residual = ut - D * uxx - r*u*(1-u) - f_term
  pde_loss = F.mse_loss(residual, torch.zeros_like(residual)).item()
  ##compute the relative PDE loss
  f_term_mean = abs(f_term).mean().item()
  residual_mean = abs(residual).mean().item()
  if f_term_mean == 0:
      relative_pde_loss = 'NaN'
  else:
      relative_pde_loss = residual_mean/f_term_mean

  ##compute the boundary loss now
  _, H0  = model(boundary_tensor)
  ##detach the H0
  H0 = H0.reshape(2*H0.shape[0], -1) #transform all H into shape (3*3*B, W)
  H0 = H0[[2*i for i in range(3*B)]] ##only pick the entries that corresponds to u
  if bias: H0 = torch.hstack((H0, torch.ones((H0.shape[0], 1))))

  ##compute the u0 = H0W
  u0 = H0@torch.tensor(W) ##(2*B, 1)
  bc_loss = F.mse_loss(u0, b0*torch.ones_like(u0)).item()
  if b0 == 0:
      relative_bc_loss = 'NaN'
  else:
      relative_bc_loss = bc_loss/b0

  result = {
      'total_loss': pde_loss + bc_loss,
      'pde_loss': pde_loss,
      'bc_loss': bc_loss, 
      'relative_pde_loss': relative_pde_loss,
      'relative_bc_loss': relative_bc_loss
  }
  return result
