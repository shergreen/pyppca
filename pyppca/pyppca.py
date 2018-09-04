# -*- coding: utf-8 -*-
"""
Python implemention of PPCA-EM for data with missing values
Adapted from MATLAB implemention from J.J. VerBeek
Modifications made in order to support high-dimensional matrices

Sheridan Beckwith Green
sheridan.green@yale.edu
Sept 2018
"""

import numpy as np
from numpy import shape, isnan, nanmean, average, zeros, log, cov
from numpy import matmul as mm
from numpy.matlib import repmat
from numpy.random import normal
from numpy.linalg import inv, det, eig
from numpy import identity as eye
from numpy import trace as tr

def ppca(Y,d,dia):
    """
    Implements probabilistic PCA for data with missing values,
    using a factorizing distribution over hidden states and hidden observations.

    Args:
        Y:   (N by D ) input numpy ndarray of data vectors
        d:   (  int  ) dimension of latent space
        dia: (boolean) if True: print objective each step

    Returns:
        ss: ( float ) isotropic variance outside subspace
        C:  (D by d ) C*C' + I*ss is covariance model, C has scaled principal directions as cols
        M:  (D by 1 ) data mean
        X:  (N by d ) expected states
        Ye: (N by D ) expected complete observations (differs from Y if data is missing)

        Based on MATLAB code from J.J. VerBeek, 2006. http://lear.inrialpes.fr/~verbeek
    """
    N, D      = shape(Y) #N observations in D dimensions (i.e. D is number of features, N is samples)
    threshold = 1E-4 #minimal relative change in objective function to continue
    hidden    = isnan(Y)
    missing   = hidden.sum()

    if(missing > 0):
        M = nanmean(Y, axis=0)
    else:
        M = average(Y, axis=0)

    Ye = Y - repmat(M,N,1)

    if(missing > 0):
        Ye[hidden] = 0

    #initialize
    C = normal(loc=0.0, scale=1.0, size=(D,d))
    CtC = mm(C.T, C)
    X   = mm(mm(Ye, C), inv(CtC))
    recon = mm(X, C.T)
    recon[hidden] = 0
    ss = np.sum((recon-Ye)**2) / (N*D - missing)

    count = 1
    old = np.inf

    #EM Iterations
    while(count):
        Sx = inv(eye(d) + CtC/ss) #E-step, covariances
        ss_old = ss
        if(missing > 0):
            proj = mm(X,C.T)
            Ye[hidden] = proj[hidden]
        
        X = mm(mm(Ye, C), Sx/ss) #E-step: expected values

        SumXtX = mm(X.T, X) #M-step
        C = mm(mm(mm(Ye.T, X),(SumXtX + N*Sx).T), inv(mm((SumXtX + N*Sx), (SumXtX + N*Sx).T)))
        CtC = mm(C.T, C)
        ss = (np.sum((mm(X, C.T) - Ye)**2) + N*np.sum(CtC*Sx) + missing*ss_old) / (N*D)
        #transform Sx determinant into numpy float128 in order to deal with high dimensionality
        Sx_det = np.min(Sx).astype(np.float128)**shape(Sx)[0] * det(Sx / np.min(Sx))
        objective = N*D + N*(D*log(ss)+tr(Sx)-log(Sx_det)) + tr(SumXtX) - missing*log(ss_old)
        
        rel_ch = np.abs( 1 - objective / old )
        old = objective
        
        count = count + 1
        if( rel_ch < threshold and count > 5 ):
            count = 0
        if(dia == True):
            print('Objective: %.2f, Relative Change %.5f' %(objective, rel_ch))
    
    C = orth(C)
    covM = cov(mm(Ye, C).T)
    vals, vecs = eig( covM )
    ordr = np.argsort(vals)[::-1]
    vals = vals[ordr]
    vecs = vecs[:,ordr]
    
    C = mm(C, vecs)
    X = mm(Ye, C)
    
    #add data mean to expected complete data
    Ye = Ye + repmat(M,N,1)

    return C, ss, M, X, Ye
