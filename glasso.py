# glasso.py
import block as bk
import numpy as np
import time
import multiprocessing as mp
import sys

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

    norm_error = [epsilon+100]
    iteration = 1

    while norm_error[-1] >= epsilon:

      iter_start_time = time.time()
      last_sigma = np.copy(self.Sigma)

      for j in range(0,len(self.Sigma)):

        print("function j =", j)
        self.__updateTheta_Inv(j = j) # UpdateTheta in R
        self.__updateTheta(j = j, gamma = gamma, epsilon = epsilon) # Algorithm_3 in R
        # sys.exit()

      print('Iteration {} Time: {}'.format(iteration, time.time() - iter_start_time))
      norm_error.append(0.000001)

    print('Time: {}'.format(time.time() - fglasso_start_time))

  def __updateTheta_Inv(self, j):

    sig_small_transpose = np.delete(self.Sigma, obj = j, axis = 0)[:,j]
    sig_small_transpose = np.transpose(sig_small_transpose, (0, 2, 1))
    
    for rowblock in range(0,self.p-1):
      if rowblock != j:
        for colblock in range(0,self.p-1):
          if colblock != j:

            inner_mult = np.matmul(self.Sigma[rowblock,j], np.linalg.inv(self.Sigma[j,j]))
            self.Theta_inv[rowblock,colblock] = self.Sigma[rowblock,colblock] - np.matmul(inner_mult, sig_small_transpose[colblock])

  def __updateTheta(self, j, gamma, epsilon):

    iteration = 1
    error = [epsilon+10**3]

    while error[-1] >= epsilon:
      last_w = np.delete(self.Theta, obj = j, axis = 0)[:,j]
      theta_nj = np.delete(self.Theta_inv, obj = j, axis = 0) # remove row j
      theta_nj = np.delete(theta_nj, obj = j, axis = 1) # remove column j
      theta_nj_transpose = np.transpose(theta_nj, (1, 0, 3, 2)) # transpose theta_nj

      for k in range(0,self.p-1):
        block_residual = np.zeros((self.M,self.M))
        for l in range(0,self.p-1):
          if k != l:
            w_block = np.delete(self.Theta, obj = j, axis = 0)[l,j]
            inner_mult = np.matmul(theta_nj_transpose[l,k], w_block)
            block_residual += np.matmul(inner_mult, self.CorMat[j,j])
        fNorm = np.linalg.norm(block_residual, ord = 'fro')
        if fNorm <= gamma:
          # set theta block to 0
          pass
        else:
          # solve system of equations for theta block
          pass
      # error.append(error[-1]-50)





  
