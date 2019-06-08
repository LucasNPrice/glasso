# glasso.py
import block as bk
import numpy as np
import time
import multiprocessing as mp
import sys
from scipy.optimize import fsolve
from sklearn.metrics import confusion_matrix

class FGLasso():

  def __init__(self, CorMat, p, M):

    self.p = p
    self.M = M

    self.CorMat = bk.block_matrix(CorMat, p, M)
    self.Theta = bk.block_diag(p,M)
    self.Sigma = bk.block_diag(p,M)
    self.Theta_inv = bk.block_diag(p,M)

  def fglasso(self, gamma, epsilon):

    fglasso_start_time = time.time()
    iteration_error = [epsilon+100]
    iteration = 1

    while iteration_error[-1] >= epsilon:

      iter_start_time = time.time()
      last_sigma = np.copy(self.Sigma)

      for j in range(0,len(self.Sigma)):

        print("function j =", j)
        self.__updateTheta_Inv(j = j) # UpdateTheta in R
        self.__updateTheta(j = j, gamma = gamma, epsilon = epsilon) # Algorithm_3 in R
        self.__updateSigma(j = j)
        
      iteration_error.append(self.__getNormError(last_sigma = last_sigma))
      print('Iteration {} Error: {}'.format(iteration, iteration_error[-1]))
      print('Iteration {} Time: {}'.format(iteration, time.time() - iter_start_time))
      iteration += 1

    print('Time to Convergence: {}'.format(time.time() - fglasso_start_time))
    edges = self.getEdges(blocked_matrix = self.Theta)
    
    return(edges)

  def __updateTheta_Inv(self, j):

    sig_small_transpose = np.delete(self.Sigma, obj = j, axis = 0)[:,j]
    sig_small_transpose = np.transpose(sig_small_transpose, (0, 2, 1))
    
    for rowblock in range(0,self.p-1):
      if rowblock != j:
        for colblock in range(0,self.p-1):
          if colblock != j:

            inner_matmul = np.matmul(self.Sigma[rowblock,j], np.linalg.inv(self.Sigma[j,j]))
            self.Theta_inv[rowblock,colblock] = self.Sigma[rowblock,colblock] - np.matmul(inner_matmul, sig_small_transpose[colblock])

  def __updateTheta(self, j, gamma, epsilon):

    iteration = 1
    error = [epsilon+10**3]

    while error[-1] >= epsilon:
      last_w = np.delete(self.Theta, obj = j, axis = 0)[:,j]
      theta_inv_nj = np.delete(self.Theta_inv, obj = j, axis = 0) # remove row j
      theta_inv_nj = np.delete(theta_inv_nj, obj = j, axis = 1) # remove column j
      theta_inv_nj_transpose = np.transpose(theta_inv_nj, (1, 0, 3, 2)) # transpose theta_inv_nj

      for k in range(0,self.p-1):
        block_residual = np.zeros((self.M,self.M))
        for l in range(0,self.p-1):
          if k != l:
            w_block = np.delete(self.Theta, obj = j, axis = 0)[l,j]
            inner_mult = np.matmul(theta_inv_nj_transpose[l,k], w_block)
            block_residual += np.matmul(inner_mult, self.CorMat[j,j])

        fNorm = np.linalg.norm(block_residual, ord = 'fro')
        wj = self.Theta[0:self.p-1,j]

        if fNorm <= gamma:
          # set theta block to 0
          wj[k] = np.zeros(self.M)
        else:
          # solve system of equations for theta block
          wj[k] = fsolve(self.__wjk, x0 = np.identity(self.M), 
            args = (theta_inv_nj, block_residual, gamma, j, k))

      iter_j = np.delete(np.arange(0,self.p,1), obj = j)
      self.Theta[iter_j,j] = wj
      self.Theta[j,iter_j] = np.transpose(wj,(0,2,1))
      iter_difference = np.reshape(self.Theta[iter_j,j],(49*5,5)) - np.reshape(last_w,(49*5,5))
      error.append(np.linalg.norm(iter_difference, ord = 'fro'))
      iteration += 1

  def __wjk(self, x, theta_inv_nj, block_residual, gamma, j, k):

    w = np.reshape(x, (self.M, self.M))
    kron_prod = np.kron(theta_inv_nj[k,k], self.CorMat[j,j])
    wj = np.matmul(kron_prod, w.flatten()) + block_residual.flatten() + (
      gamma * (w.flatten() / np.linalg.norm(w, ord = 'fro')))
    return(wj)

  def __updateSigma(self, j):

    innerblock = np.zeros((self.M,self.M))
    Uj = np.reshape(np.repeat(innerblock, self.p-1), (49,5,5))
    theta_inv_nj = np.delete(self.Theta_inv, obj = j, axis = 0) # remove row j
    theta_inv_nj = np.delete(theta_inv_nj, obj = j, axis = 1) # remove col j
    theta_nj = np.delete(self.Theta_inv, obj = j, axis = 0) # remove row j

    for rowblock in range(0,len(Uj)):
      for colblock in range(0,len(Uj)):
        Uj[colblock] += np.matmul(theta_inv_nj[rowblock,colblock], theta_nj[colblock,j])

    Uj_transpose = np.transpose(Uj, (0,2,1))
    self.Sigma[j,j] = self.CorMat[j,j]
    iter_j = np.delete(np.arange(0,self.p,1), obj = j)

    for block in enumerate(iter_j):
      self.Sigma[block[1],j] = -1*(np.matmul(Uj[block[0]],self.CorMat[j,j]))
      self.Sigma[j,block[1]] = np.transpose(self.Sigma[block[1],j], (1,0))

    for rowblock in enumerate(iter_j):
      for colblock in enumerate(iter_j):
        inner_matmul = np.matmul(Uj[rowblock[0]], self.CorMat[j,j])
        self.Sigma[rowblock[1],colblock[1]] = self.Theta_inv[rowblock[1],colblock[1]] + (
          np.matmul(inner_matmul, Uj_transpose[colblock[0]]))

  def __getNormError(self, last_sigma):

    unblock_new_sigma = np.reshape(self.Sigma, (self.p*self.M,self.p*self.M))
    unblock_last_sigma = np.reshape(last_sigma, (self.p*self.M,self.p*self.M))
    iter_difference = unblock_new_sigma - unblock_last_sigma
    fNorm = np.linalg.norm(iter_difference, ord = 'fro')
    return(fNorm)

  def getEdges(self, blocked_matrix):

    edges = np.zeros((self.p,self.p))
    for rowblock in range(0,self.p):
      for colblock in range(0,self.p):
        if np.all(self.Theta[rowblock,colblock] == 0):
          edges[rowblock,colblock] = 0
        else:
          edges[rowblock,colblock] = 1
    return(edges)

  def getErrors(estEdges, actEdges):

    tn, fp, fn, tp = confusion_matrix(
      actEdges.flatten(), 
      estEdges.flatten()).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return((tpr,fpr))










  
