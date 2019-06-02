# glasso.py
import block as bk
import numpy as np
import time
import multiprocessing as mp
import sys

class FGLasso():

  def __init__(self, Cov, p, M):

    self.p = p
    self.M = M

    self.Cov = bk.block_matrix(Cov, p, M)
    self.Theta = bk.block_diag(p,M)
    self.Sigma = bk.block_diag(p,M)
    self.Theta_inv = bk.block_diag(p,M)

  def fglasso(self, epsilon):

    fglasso_start_time = time.time()

    norm_error = [epsilon+100]
    iteration = 1

    while norm_error[-1] >= epsilon:

      iter_start_time = time.time()
      last_sigma = np.copy(self.Sigma)

      for j in range(0,len(self.Sigma)):

        print("function j =", j)
        self.__updateTheta_Inv(j = j) # UpdateTheta in R
        self.__updateTheta(j = j, epsilon = epsilon) # Algorithm_3 in R
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

  def __updateTheta(self, j, epsilon):

    iteration = 1
    error = [epsilon+10**3]

    while error[-1] >= epsilon:
      last_w = np.delete(self.Theta, obj = j, axis = 0)[:,j]
      # theta_nj_transpose = 

      for k in range(0,self.p-1):
        block_residual = np.zeros((self.M,self.M))
        for l in range(0,self.p-1):
          continue

      error.append(error[-1]-50)





  
