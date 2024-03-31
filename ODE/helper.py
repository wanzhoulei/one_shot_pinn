#import all required packages
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp

from neurodiffeq import diff
from neurodiffeq.neurodiffeq import unsafe_diff as unsafe_diff
from neurodiffeq.conditions import IVP, DirichletBVP, DirichletBVP2D, BundleIVP, BundleDirichletBVP
from neurodiffeq.solvers import Solver1D, Solver2D, BundleSolver1D
from neurodiffeq.networks import FCNN, SinActv
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.generators import Generator1D, Generator2D

import time
from tqdm.auto import tqdm


#utility function plot the training trace
def plot_trace(loss_trace, ode_trace, ic_trace):
  ##plot the loss value
  fig, ax = plt.subplots(1, 2, figsize=(15, 5))
  num_iter = len(loss_trace)
  ax[0].set_yscale('log');
  ax[0].plot(range(1, num_iter+1), loss_trace, label='Total Loss');

  ax[1].set_yscale('log');
  ax[1].plot(range(1, num_iter+1), ode_trace, label='ODE Loss');
  ax[1].plot(range(1, num_iter+1), ic_trace, label="Initial Condition Loss");

  ax[0].set_xlabel("Number of iterations");
  ax[0].set_ylabel("Loss");
  ax[0].set_title("Total Loss Value vs. Iteration");
  ax[0].grid();
  ax[0].legend();
  ax[1].set_xlabel("Number of iterations");
  ax[1].set_ylabel("Loss");
  ax[1].set_title("ODE and IC Loss Value vs. Iteration");
  ax[1].grid();
  ax[1].legend();
  
##define the forcing functions
def forcing_decorator(gamma, omega):
  def force(t):
    return gamma*torch.cos(omega*t)
  return force

##this function trains the multi-headed neural network
def train(model, optimizer, lossfn, num_iter, para_dict, sample_size=100, domain=(0, 5), every=100, ode_weight=1,
          ic_weight=1, scheduler=None, epsilon=1e-3, dtype=torch.float32, verbose=True):
  loss_trace = []; ode_loss_trace = []; ic_loss_trace = [];
  Forcing_functions = para_dict['Forcing_functions']
  initial_values = para_dict['initial_values']
  initial_velocities = para_dict['initial_velocities']
  delta_list = para_dict['delta_list']
  alpha_list = para_dict['alpha_list']
  for i in tqdm(range(num_iter)):
    optimizer.zero_grad()
    #evaluate the loss
    loss_dict = lossfn(model, Forcing_functions, delta_list, alpha_list, initial_values, initial_velocities, domain=domain, sample_size=sample_size,
         epsilon=epsilon, alpha_ode = ode_weight, alpha_ic = ic_weight, dtype=dtype)
    #take the step
    loss_dict['total_loss'].backward()
    optimizer.step()
    if scheduler is not None:
      scheduler.step()
    #record
    loss_trace.append(loss_dict['total_loss'].item())
    ode_loss_trace.append(loss_dict['ode_loss'].item())
    ic_loss_trace.append(loss_dict['ic_loss'].item())
    #print
    if verbose and (i+1)%every == 0:
      print("{}th Iter: total {}, ode {}, ic {}".format(i+1, loss_dict['total_loss'].item(),
                                      loss_dict['ode_loss'].item(), loss_dict['ic_loss'].item()))
  return loss_trace, ode_loss_trace, ic_loss_trace
  
def generate_parameters(k, gamma_domain = (0.5, 3), omega_domain = (0.5, 3), alpha_domain = (0.5, 4.5), delta_domain = (0.5, 4.5), 
                        initial_domain = (-3, 3), velocity_domain = (-1, 1), seed = 42):
    np.random.seed(seed)
    #generate these random parameters to use in the k heads
    gamma_list = np.random.uniform(*gamma_domain, k)
    omega_list = np.random.uniform(*omega_domain, k)
    alpha_list = np.random.uniform(*alpha_domain, k)
    delta_list = np.random.uniform(*delta_domain, k)
    initial_values = np.random.uniform(*initial_domain, k)
    initial_velocities = np.random.uniform(*velocity_domain, k)

    #construct the forcing functions list
    forcing_list = []
    for i in range(k):
      forcing_list.append(forcing_decorator(gamma_list[i], omega_list[i]))

    return {'gamma_list': gamma_list, 'omega_list': omega_list, 'alpha_list': alpha_list, 
            'delta_list': delta_list, 'initial_values': initial_values, 'initial_velocities': initial_velocities, 
            'Forcing_functions': forcing_list}

def RHS_decorator(gamma, w, alpha, delta):
  def func(t, y):
    y = np.array(y)
    A_mat = np.array([[0, -1], [alpha, delta]])
    return -A_mat@y + np.array([0, gamma*np.cos(w*t)])
  return func
  
def compute_MSE(NN_sol, numerical_sol):
  NN_sol = np.array([NN_sol[i].detach().numpy().T for i in range(NN_sol.shape[0])])
  numerical_sol = np.array([ele.y for ele in numerical_sol])
  return ((NN_sol - numerical_sol)**2).mean()
  
##function to compute the gradient of H w.r.t. xt
def compute_Ht(H, t):
  output = []
  for i in range(H.shape[1]):
    output.append(diff(H[:,i].reshape(-1, 1), t).detach().numpy())
  return np.concatenate(output, axis=1)

##given 2 by 2 matrix A and matrix H of shape (2N, 257)
##compute the matrix AH of dimension (2N, 257)
def compute_AH(A, H):
  A_reshaped = A.reshape(1, 2, 2)
  #reshape H to (N, 2, 257)
  H_reshaped = H.reshape(-1, 2, H.shape[-1])
  # Perform the multiplication
  AH = np.matmul(A_reshaped, H_reshaped)
  # Reshape the result back
  AH = AH.reshape(-1, AH.shape[-1])
  return AH

#this function computes and returns M and its inverse
#it takes inputs: H, Ht, AH, H0
def compute_M(H, Ht, AH, H0):
  N = H.shape[0]/2
  HtAH = (Ht.T @ AH)
  HAHt = (AH).T @ Ht
  HtHt = Ht.T @ Ht
  HAAH = (AH).T @ AH

  H0tH0 = H0.T @ H0

  M = (HtAH + HAHt + HtHt + HAAH)/N + H0tH0
  Minv = np.linalg.pinv(M)
  return M, Minv

##this function computes and returns the transfer learning W
def compute_TLW(f, initial_value, initial_velocity, Ht, AH, H0, Minv, t_grids):
  u0 = np.array([[initial_value], [initial_velocity]])
  f_values = f(t_grids).detach().numpy()
  F = np.concatenate([np.zeros((f_values.T).shape), f_values.T]).T.reshape(-1, 1)
  N = t_grids.shape[0]
  R = H0.T @ u0 + (Ht.T @ F)/N + (AH.T @ F)/N
  W = Minv @ R
  return W

##this function computes the TL solution for an unseen linear ODE system
##H is the hidden state of the NN at specified grid points, if it is None, it needs to be computed
#H should have shape (2N, 257)
##t_grids is a 1d numpy array that specifies the grid points that needed to be evaluated, 
#it can only be none if H is not None
def compute_TLsolution(W, H = None, t_grids = None, model=None, dtype=torch.float32):
  if H is None and t_grids is None:
    raise ValueError("H and t_grids cannot both be None");
  result_dict = {}
  #if the user specfies the hidden state
  if H is not None:
    result = (H @ W).reshape(-1, 2)
    result_dict['x'] = result[:,0]
    result_dict['y'] = result[:,1]
  else: #if it is none, we need to compute it
    if model is None: raise ValueError("model cannot be None");
    t_grids = torch.tensor(t_grids, dtype=dtype).view(-1, 1)
    _, H = model(t_grids)
    H = np.hstack((H.reshape(2*H.shape[0], -1).detach().numpy(), np.ones((2*H.shape[0], 1))))
    result = (H @ W).reshape(-1, 2)
    result_dict['x'] = result[:,0]
    result_dict['y'] = result[:,1]
  return result_dict

##this function computes the solution of an unseen linear ODE system
#f: the function R --> R handler of the forcing function 
#alpha, delta the parameter values of the new linear ODE system
#initial_value, initial_velocity, the initial value and velovity of the new system
#H, Ht, H0 matrices computed to use to compute the M and Minv matrices
def One_Shot_solve(f, H, Ht, H0, alpha, delta, initial_value, initial_velocity, t_grids):
  A = np.array([[0, -1], [alpha, delta]])
  AH = compute_AH(A, H)
  M, Minv = compute_M(H, Ht, AH, H0)
  W = compute_TLW(f, initial_value, initial_velocity, Ht, AH, H0, Minv, t_grids)
  result = compute_TLsolution(W, H)
  result['W'] = W
  return result

##this function computes the linear combination of a list of weights W
## W = \sum_i=0^p beta^i W_i
def combine_W(W_list, beta):
  W = W_list[0]
  for i in range(1, len(W_list)):
    W += (beta**i)*W_list[i]
  return W

##this function compute the boundary value for each function
def compute_each_iv(iv, beta, p):
  deno = 1
  for i in range(1, p+1):
    deno += beta**i
  return iv/deno

##this function computes and returns the transfer learning W from forcing function values
def compute_TLW_fromfvalues(f_values, initial_value, initial_velocity, Ht, AH, H0, Minv):
  u0 = np.array([[initial_value], [initial_velocity]])
  F = np.concatenate([np.zeros((f_values.T).shape), f_values.T]).T.reshape(-1, 1)
  N = f_values.shape[0]
  R = H0.T @ u0 + (Ht.T @ F)/N + (AH.T @ F)/N
  W = Minv @ R
  return W

from itertools import product

def find_three_integers(target):
    nums = list(range(target+1))
    # Generate all combinations of three integers with repetition from the list
    all_combinations = list(product(nums, repeat=3))

    # Filter combinations that sum to the target
    valid_combinations = [combo for combo in all_combinations if (sum(combo) == target and combo[0]<=combo[1] and combo[1]<=combo[2])]

    return valid_combinations

##this function computes the linear combination of a list of weights W
## W = \sum_i=0^p lambda^i W_i
def combine_W(W_list, beta):
  W = W_list[0].copy()
  for i in range(1, len(W_list)):
    W += (beta**i)*W_list[i]
  return W

##this function conduct p-shots TL to solve a non linear ODE using perturbation method
def p_shot_solve(f0_values, beta, alpha, delta, initial_value, initial_velocity, H, Ht, H0, indices_list, p=12):
    A = np.array([[0, -1], [alpha, delta]])
    AH = compute_AH(A, H)
    M, Minv = compute_M(H, Ht, AH, H0)
    ##set initial conditions for all p+1 ODE systems
    ini_val = compute_each_iv(initial_value, beta, p)
    ini_vel = compute_each_iv(initial_velocity, beta, p)
    W_list = []; x_list = []; f_list = []; y_list = []
    #compute the oth ode
    W0 = compute_TLW_fromfvalues(f0_values, ini_val, ini_vel, Ht, AH, H0, Minv)
    u0 = compute_TLsolution(W0, H) 
    W_list.append(W0); x_list.append(u0['x']); f_list.append(f0_values); y_list.append(u0['y'])
    
    ##solve and record all the remaining p PDEs
    for i in range(1, p+1):
        target = i-1
        indices = indices_list[i-1]
        fi_values = np.zeros(x_list[0].shape)
        for combo in indices:
            if combo[0] != combo[1]:
                if combo[1] != combo[2]:
                    fi_values -= 6*x_list[combo[0]]*x_list[combo[1]]*x_list[combo[2]]
                else:
                    fi_values -= 3*x_list[combo[0]]*x_list[combo[1]]*x_list[combo[2]]
            else:
                if combo[1] != combo[2]:
                    fi_values -= 3*x_list[combo[0]]*x_list[combo[1]]*x_list[combo[2]]
                else:
                    fi_values -= x_list[combo[0]]*x_list[combo[1]]*x_list[combo[2]]
        #reshape fi_values into (n, 1)
        fi_values = fi_values.reshape(-1, 1)
        #solve and record the ith ode
        Wi = compute_TLW_fromfvalues(fi_values, ini_val, ini_vel, Ht, AH, H0, Minv)
        ui = compute_TLsolution(Wi, H) 
        W_list.append(Wi); x_list.append(ui['x']); f_list.append(fi_values); y_list.append(ui['y'])
    #compute the final W
    W = combine_W(W_list, beta)
    #compute the solution 
    x = combine_W(x_list, beta)
    y = combine_W(y_list, beta)
    dict = {
        'W': W, 'x': x, 'y': y, 'W_list': W_list, 'x_list': x_list, 'y_list': y_list, 'f_list': f_list
    }
    return dict

##this function solves a Duffing Equation numerically
##f is a 1-1 function w.r.t. t
def solve_duffing(delta, alpha, beta, f, u0, domain, t_eval):
  def F(t, y):
    return [y[1],
            -delta*y[1]- alpha*y[0] - beta*y[0]**3 + f(t)]
  solution = solve_ivp(F, domain, u0, t_eval=t_eval, method='DOP853')
  return solution

def compute_duffing_loss(H, Ht, Htt, H0, W, delta, alpha, beta, f_value, u0, v0):
    H_total = Htt + delta*Ht + alpha*H #shape (2N, 257)
    LHS = H_total @ W #shape (2N, 1)
    ##obtain the x
    N = int(LHS.shape[0]/2)
    x = LHS[range(0, 2*N, 2)].flatten() + beta*((H@W)[range(0, 2*N, 2)].flatten())**3
    f_value = f_value.flatten()
    ode_loss = ((x - f_value)**2).mean()
    ##compute IC loss
    model_initials = H0@W.flatten()
    initials = np.array([u0, v0])
    IC_loss = ((model_initials - initials)**2).mean()
    return {
        'total_loss': IC_loss + ode_loss, 'ode_loss': ode_loss, 'ic_loss': IC_loss
    }