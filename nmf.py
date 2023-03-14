#!/usr/bin/python

import numpy as np
from pandas import DataFrame

def random_initialization(A,rank):
    """Random initialization of W and H matrices.
    Parameters
    ----------
    A : array-like, shape (n_samples, n_features)
        The input data.
    rank : int
        The rank of the factorization.
    Returns
    -------
    W : array-like, shape (n_samples, rank)
        Initial guesses for solving X ~= WH
    H : array-like, shape (rank, n_features)
        Initial guesses for solving X ~= WH

    """
    number_of_documents = A.shape[0]
    number_of_terms = A.shape[1]
    W = np.random.uniform(1,2,(number_of_documents,rank))
    H = np.random.uniform(1,2,(rank,number_of_terms))
    return W,H
                          

def nndsvd_initialization(A,rank):
    """NNDSVD initialization of W and H matrices.
    Parameters
    ----------
    A : array-like, shape (n_samples, n_features)
        The input data.
    rank : int
        The rank of the factorization.
    Returns
    -------
    W : array-like, shape (n_samples, rank)
        Initial guesses for solving X ~= WH
    H : array-like, shape (rank, n_features)
        Initial guesses for solving X ~= WH

    """
    u,s,v=np.linalg.svd(A,full_matrices=False)
    v=v.T
    w=np.zeros((A.shape[0],rank))
    h=np.zeros((rank,A.shape[1]))

    w[:,0]=np.sqrt(s[0])*np.abs(u[:,0])
    h[0,:]=np.sqrt(s[0])*np.abs(v[:,0].T)

    for i in range(1,rank):
        
        ui=u[:,i]
        vi=v[:,i]
        ui_pos=(ui>=0)*ui
        ui_neg=(ui<0)*-ui
        vi_pos=(vi>=0)*vi
        vi_neg=(vi<0)*-vi
        
        ui_pos_norm=np.linalg.norm(ui_pos,2)
        ui_neg_norm=np.linalg.norm(ui_neg,2)
        vi_pos_norm=np.linalg.norm(vi_pos,2)
        vi_neg_norm=np.linalg.norm(vi_neg,2)
        
        norm_pos=ui_pos_norm*vi_pos_norm
        norm_neg=ui_neg_norm*vi_neg_norm
        
        if norm_pos>=norm_neg:
            w[:,i]=np.sqrt(s[i]*norm_pos)/ui_pos_norm*ui_pos
            h[i,:]=np.sqrt(s[i]*norm_pos)/vi_pos_norm*vi_pos.T
        else:
            w[:,i]=np.sqrt(s[i]*norm_neg)/ui_neg_norm*ui_neg
            h[i,:]=np.sqrt(s[i]*norm_neg)/vi_neg_norm*vi_neg.T

    return w,h
def mu_method(A,k,max_iter,init_mode='random'):
    """Multiplicative update method for non-negative matrix factorization.
    Parameters
    ----------
    A : array-like, shape (n_samples, n_features)
        The input data.
    k : int
        The rank of the factorization.
    max_iter : int
        The maximum number of iterations.
    init_mode : string, default 'random'
        The initialization mode. It can be 'random' or 'nndsvd'.
    Returns
    -------
    W : array-like, shape (n_samples, rank)
        Initial guesses for solving X ~= WH
    H : array-like, shape (rank, n_features)
    
    """
    
    if init_mode == 'random':
        W ,H = random_initialization(A,k)
    elif init_mode == 'nndsvd':
        W ,H = nndsvd_initialization(A,k) 
    norms = []
    e = 1.0e-10
    for n in range(max_iter):
        # Update H
        # W_TA = W.T@A
        # W_TWH = W.T@W@H+e
        # for i in range(np.size(H, 0)):
        #     for j in range(np.size(H, 1)):
        #         H[i, j] = H[i, j] * W_TA[i, j] / W_TWH[i, j]
        # vectorized version
        W_TA = W.T @ A
        W_TWH = W.T @ W @ H + e
        H = H * W_TA / W_TWH        
        # Update W
        # AH_T = A@H.T
        # WHH_T =  W@H@H.T+ e

        # for i in range(np.size(W, 0)):
        #     for j in range(np.size(W, 1)):
        #         W[i, j] = W[i, j] * AH_T[i, j] / WHH_T[i, j]
        # vectorized version
        AH_T = A @ H.T
        WHH_T = W @ H @ H.T + e
        W = W * AH_T / WHH_T

        norm = np.linalg.norm(A - W@H, 'fro')
        norms.append(norm)
        print('Iteration: ', n, 'norm: ', norm)
    return W ,H ,norms 