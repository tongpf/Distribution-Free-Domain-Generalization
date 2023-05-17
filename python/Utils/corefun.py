import copy
import numpy as np
from scipy.linalg import inv
import scipy


def fast_kernel(X, Y, bandwidth):
    sq_X = (X ** 2).sum(axis=1)
    sq_Y = (Y ** 2).sum(axis=1)
    dists_sq = np.dot(X, Y.T)
    gram_matrix = -sq_X[:, None] - sq_Y[None, :] + 2 * dists_sq
    if bandwidth == 'auto':
        gram_matrix = np.exp(gram_matrix / X.shape[1])
    else:
        gram_matrix = np.exp(gram_matrix / bandwidth)
    return gram_matrix


def transfer_kernel(X, Y, Xlabel, Ylabel, bandwidth='auto'):
    xclasslist = np.unique(Xlabel)
    yclasslist = np.unique(Ylabel)
    gram_matrix = np.ones((X.shape[0], Y.shape[0]))
    class_matrix = _transfer_kernel(X, Y, Xlabel, Ylabel, bandwidth)
    for xclass_idx in range(len(xclasslist)):
        for yclass_idx in range(len(yclasslist)):
            xclass = xclasslist[xclass_idx]
            yclass = yclasslist[yclass_idx]
            idxx = (Xlabel == xclass)
            idxy = (Ylabel == yclass)
            subX = X[idxx, :]
            subY = Y[idxy, :]
            subgram_matrix = fast_kernel(subX, subY, bandwidth)
            gram_matrix[np.ix_(idxx, idxy)] = subgram_matrix * class_matrix[xclass_idx, yclass_idx]
    return gram_matrix


def _transfer_kernel(X, Y, Xlabel, Ylabel, bandwidth):
    xclasslist = np.unique(Xlabel)
    yclasslist = np.unique(Ylabel)
    class_matrix = np.ones((len(xclasslist), len(yclasslist)))
    for xclass_idx in range(len(xclasslist)):
        for yclass_idx in range(len(yclasslist)):
            xclass = xclasslist[xclass_idx]
            yclass = yclasslist[yclass_idx]
            if xclass != yclass:
                idxx = (Xlabel == xclass)
                idxy = (Ylabel == yclass)
                subX = X[idxx, :]
                subY = Y[idxy, :]
                shrinkcoeff = np.mean(fast_kernel(subX, subX, bandwidth)) + np.mean(fast_kernel(subY, subY, bandwidth)) - \
                              2 * np.mean(fast_kernel(subX, subY, bandwidth))
                shrinkcoeff = np.exp(-shrinkcoeff/2)
                class_matrix[xclass_idx, yclass_idx] = shrinkcoeff
    return class_matrix


def DFDG_Eig(X, Y, Xlabel, bandwidth='auto', qmax='auto', epsilon=1e-5, gamma=1):

    xclasslist = np.unique(Xlabel)
    yclasslist = np.unique(Y)
    n = X.shape[0]
    n_domain = len(xclasslist)
    n_class = len(yclasslist)

    # the Gram matrix
    Kmatrix = transfer_kernel(X, X, Xlabel, Xlabel, bandwidth=bandwidth)
    Kmatrix = Kmatrix.astype(np.float64)

    Kmatrix_raw = copy.deepcopy(Kmatrix)
    # centering Gram matrix
    Imatrix = np.ones((n, n))/n
    Kmatrix = Kmatrix - Imatrix.dot(Kmatrix) - Kmatrix.dot(Imatrix) + Imatrix.dot(Kmatrix).dot(Imatrix)

    if qmax == 'auto':
        qmax = int(n * 0.05)

    # scaled pairwise cross class discrepancy
    Fmatrix = np.zeros((n, n))
    for idomain_idx in range(n_domain):
        idomain = xclasslist[idomain_idx]
        idxi = (Xlabel == idomain)
        for iclass_idx in range(n_class):
            iclass = yclasslist[iclass_idx]
            idxj = (Y[idxi] == iclass)
            for iclass_idx2 in range(iclass_idx+1, n_class):
                iclass2 = yclasslist[iclass_idx2]
                idxk = (Y[idxi] == iclass2)
                nj = sum(idxj)
                nk = sum(idxk)
                Kmatrix_j = Kmatrix[:, idxi][:, idxj].mean(axis=1)
                Kmatrix_k = Kmatrix[:, idxi][:, idxk].mean(axis=1)
                Kmatrix_diff = Kmatrix_j - Kmatrix_k
                Kmatrix_jk = Kmatrix[np.ix_(idxi, idxi)][np.ix_(idxj, idxk)]
                Imatrixj = np.ones((nj, nj)) / nj
                Imatrixk = np.ones((nk, nk)) / nk
                Kmatrix_jk = Kmatrix_jk - Imatrixj.dot(Kmatrix_jk) - Kmatrix_jk.dot(Imatrixk) + \
                          Imatrixj.dot(Kmatrix_jk).dot(Imatrixk)
                Kmatrix_jk_norm = np.linalg.norm(Kmatrix_jk, 'nuc')
                Fmatrix = Fmatrix + np.outer(Kmatrix_diff, Kmatrix_diff) / Kmatrix_jk_norm * nj * nk / (nj + nk)

    # scaled pairwise cross domain discrepancy
    Qmatrix = np.zeros((n, n))
    for iclass_idx in range(n_class):
        iclass = yclasslist[iclass_idx]
        idxi = (Y == iclass)
        for idomain_idx in range(n_domain):
            idomain = xclasslist[idomain_idx]
            idxj = (Xlabel[idxi] == idomain)
            for idomain_idx2 in range(idomain_idx+1, n_domain):
                idomain2 = xclasslist[idomain_idx2]
                idxk = (Xlabel[idxi] == idomain2)
                nj = sum(idxj)
                nk = sum(idxk)
                Kmatrix_j = Kmatrix[:, idxi][:, idxj].mean(axis=1)
                Kmatrix_k = Kmatrix[:, idxi][:, idxk].mean(axis=1)
                Kmatrix_diff = Kmatrix_j - Kmatrix_k
                Kmatrix_jk = Kmatrix[np.ix_(idxi, idxi)][np.ix_(idxj, idxk)]
                Imatrixj = np.ones((nj, nj)) / nj
                Imatrixk = np.ones((nk, nk)) / nk
                Kmatrix_jk = Kmatrix_jk - Imatrixj.dot(Kmatrix_jk) - Kmatrix_jk.dot(Imatrixk) + \
                          Imatrixj.dot(Kmatrix_jk).dot(Imatrixk)
                Kmatrix_jk_norm = np.linalg.norm(Kmatrix_jk, 'nuc')
                Qmatrix = Qmatrix + np.outer(Kmatrix_diff, Kmatrix_diff) / Kmatrix_jk_norm * nj * nk / (nj + nk)

    Qmatrix = Qmatrix + gamma * Kmatrix_raw + epsilon * np.eye(n)

    # generalized eigenvalue problem
    w, Bmatrix = scipy.linalg.eigh(Fmatrix, Qmatrix, subset_by_index=[n-qmax, n-1])
    w = w.real
    Bmatrix = Bmatrix.real

    # scaling B matrix
    B_norm = np.diag(Bmatrix.T @ Kmatrix_raw @ Bmatrix)
    Bmatrix = Bmatrix / (np.abs(B_norm[None, :]) ** (1 / 2))

    return Bmatrix, Kmatrix_raw, w


def DFDG_Cov(X, Y, Xlabel, bandwidth='auto', qmax='auto', epsilon=1e-5, gamma=1):
    xclasslist = np.unique(Xlabel)
    yclasslist = np.unique(Y)
    n = X.shape[0]
    n_domain = len(xclasslist)
    n_class = len(yclasslist)
    # Gram matrix
    Kmatrix = transfer_kernel(X, X, Xlabel, Xlabel, bandwidth=bandwidth)
    Kmatrix = Kmatrix.astype(np.float64)

    if qmax == 'auto':
        qmax = int(n * 0.05)

    # cross class discrepancy
    Kmatrix_pcd = np.zeros((n, n_class * n_domain))
    for idomain_idx in range(n_domain):
        idomain = xclasslist[idomain_idx]
        for iclass_idx in range(n_class):
            iclass = yclasslist[iclass_idx]
            idxj = (Y == iclass)
            idxi = (Xlabel[idxj] == idomain)
            Kmatrix_pcd[:, idomain_idx*n_class + iclass_idx] = Kmatrix[:, idxj][:, idxi].mean(axis=1)

    Gamma_pcd = np.zeros((n_class * n_domain, n_class * n_domain))
    for idomain_idx in range(n_domain):
        idomain = xclasslist[idomain_idx]
        for iclass_idx in range(n_class):
            iclass = yclasslist[iclass_idx]
            idxj = (Y == iclass)
            idxi = (Xlabel[idxj] == idomain)
            Gamma_pcd = Gamma_pcd + np.cov(Kmatrix_pcd[idxj, :][idxi, :].T)
    Gamma_pcd = np.linalg.pinv(Gamma_pcd / n_class / n_domain, hermitian=True)

    Gamma_pcd_sqrt = scipy.linalg.sqrtm(Gamma_pcd)
    Gamma_pcd = n_class * np.eye(n_class) - np.outer(np.ones(n_class), np.ones(n_class))
    Gamma_pcd = np.kron(np.eye(n_domain, dtype=int), Gamma_pcd)
    Gamma_pcd = Gamma_pcd_sqrt @ Gamma_pcd @ Gamma_pcd_sqrt

    Fmatrix = Kmatrix_pcd.dot(Gamma_pcd).dot(Kmatrix_pcd.T) / (n_class * (n_class-1) / 2) / n_domain
    Fmatrix = Fmatrix.real

    # cross domain discrepancy
    Kmatrix_pdd = np.zeros((n, n_class * n_domain))
    for iclass_idx in range(n_class):
        iclass = yclasslist[iclass_idx]
        for idomain_idx in range(n_domain):
            idomain = xclasslist[idomain_idx]
            idxi = (Xlabel == idomain)
            idxj = (Y[idxi] == iclass)
            Kmatrix_pdd[:, iclass_idx*n_domain + idomain_idx] = Kmatrix[:, idxi][:, idxj].mean(axis=1)

    Gamma_pdd = np.zeros((n_class * n_domain, n_class * n_domain))
    for iclass_idx in range(n_class):
        iclass = yclasslist[iclass_idx]
        for idomain_idx in range(n_domain):
            idomain = xclasslist[idomain_idx]
            idxi = (Xlabel == idomain)
            idxj = (Y[idxi] == iclass)
            Gamma_pdd = Gamma_pdd + np.cov(Kmatrix_pdd[idxi, :][idxj, :].T)
    Gamma_pdd = np.linalg.pinv(Gamma_pdd / n_class / n_domain, hermitian=True)

    Gamma_pdd_sqrt = scipy.linalg.sqrtm(Gamma_pdd)
    Gamma_pdd = n_domain * np.eye(n_domain) - np.outer(np.ones(n_domain), np.ones(n_domain))
    Gamma_pdd = np.kron(np.eye(n_class, dtype=int), Gamma_pdd)
    Gamma_pdd = Gamma_pdd_sqrt @ Gamma_pdd @ Gamma_pdd_sqrt

    Gmatrix = Kmatrix_pdd.dot(Gamma_pdd).dot(Kmatrix_pdd.T) / (n_domain * (n_domain-1) / 2) / n_class
    Gmatrix = Gmatrix.real

    Cmatrix = Fmatrix
    Qmatrix = Gmatrix + gamma * Kmatrix + epsilon * np.eye(n)

    # generalized eigenvalue problem
    w, Bmatrix = scipy.linalg.eigh(Cmatrix, Qmatrix, subset_by_index=[n-qmax, n-1])
    w = w.real
    Bmatrix = Bmatrix.real

    # scaling B matrix
    B_norm = np.diag(Bmatrix.T @ Kmatrix @ Bmatrix)
    Bmatrix = Bmatrix / (np.abs(B_norm[None, :]) ** (1 / 2))

    return Bmatrix, Kmatrix, w


def transfer_kernel_tong_onescaling_new(X, Y, Xlabel, bandwidth='auto', qmax='auto', epsilon=1e-4, gamma=1):
    # training_idx1 = np.random.choice(range(X.shape[0]), 5000, replace=False)
    # X = X[training_idx1]
    # Y = Y[training_idx1]
    # Xlabel = Xlabel[training_idx1]
    xclasslist = np.unique(Xlabel)
    yclasslist = np.unique(Y)
    n = X.shape[0]
    n_domain = len(xclasslist)
    n_class = len(yclasslist)
    # yclasslist = np.unique(Ylabel)
    Kmatrix = transfer_kernel(X, X, Xlabel, Xlabel, bandwidth=bandwidth)
    # Kmatrix = fast_kernel(X, X, bandwidth=bandwidth)
    Kmatrix = Kmatrix.astype(np.float64)
    # print(Kmatrix.max())
    #Kmatrix_raw = copy.deepcopy(Kmatrix)
    #Imatrix = np.ones((n, n))/n
    #Kmatrix = Kmatrix - Imatrix.dot(Kmatrix) - Kmatrix.dot(Imatrix) + \
    #          Imatrix.dot(Kmatrix).dot(Imatrix)
    # print(Kmatrix.max())
    # print(np.linalg.eigh(Kmatrix)[0])

    if qmax == 'auto':
        qmax = int(n * 0.05)

    Kmatrix_acd = np.zeros((n, n_class * n_domain))
    for idomain_idx in range(n_domain):
        idomain = xclasslist[idomain_idx]
        for iclass_idx in range(n_class):
            iclass = yclasslist[iclass_idx]
            idxj = (Y == iclass)
            idxi = (Xlabel[idxj] == idomain)
            Kmatrix_acd[:, idomain_idx*n_class + iclass_idx] = Kmatrix[:, idxj][:, idxi].mean(axis=1)
    # Gamma_acd = Kmatrix_acd[:, 0:n_class].T.dot(Kmatrix_acd[:, 0:n_class])
    # for idomain_idx in range(1, n_domain):
    #     temp_mat = Kmatrix_acd[:, (idomain_idx*n_class):((idomain_idx+1)*n_class)]
    #     Gamma_acd = scipy.linalg.block_diag(Gamma_acd, temp_mat.T.dot(temp_mat))
    # Gamma_acd = np.linalg.pinv(Gamma_acd * Kmatrix_acd.shape[1], hermitian=True)

    Gamma_acd = np.zeros((n_class * n_domain, n_class * n_domain))
    for idomain_idx in range(n_domain):
        idomain = xclasslist[idomain_idx]
        for iclass_idx in range(n_class):
            iclass = yclasslist[iclass_idx]
            idxj = (Y == iclass)
            idxi = (Xlabel[idxj] == idomain)
            Gamma_acd = Gamma_acd + np.cov(Kmatrix_acd[idxj, :][idxi, :].T)
    Gamma_acd = np.linalg.pinv(Gamma_acd / n_class / n_domain, hermitian=True)

    #Gamma_acd = np.linalg.pinv(np.cov(Kmatrix_acd.T), hermitian=True)#Kmatrix_acd.T.dot(Kmatrix_acd) #/ Kmatrix_acd.shape[0]
    Gamma_acd_sqrt = scipy.linalg.sqrtm(Gamma_acd)

    for idomain_idx in range(n_domain):
        idomain = xclasslist[idomain_idx]
        for iclass_idx in range(n_class):
            iclass = yclasslist[iclass_idx]
            idxj = (Y == iclass)
            idxi = (Xlabel[idxj] == idomain)
            Kmatrix_acd_block = Kmatrix_acd[idxj, :][idxi, :]
            Kmatrix_mean = Kmatrix_acd_block.mean(axis=0)
            Kmatrix_acd_block = Kmatrix_acd_block - Kmatrix_mean
            Kmatrix_acd_block = Kmatrix_acd_block @ Gamma_acd_sqrt
            Kmatrix_acd_block = Kmatrix_acd_block + Kmatrix_mean
            Kmatrix_acd[idxj, :][idxi, :] = Kmatrix_acd_block

    Gamma_acd = n_class * np.eye(n_class) - np.outer(np.ones(n_class), np.ones(n_class))
    Gamma_acd = np.kron(np.eye(n_domain, dtype=int), Gamma_acd)
    #Gamma_acd = Gamma_acd_sqrt @ Gamma_acd @ Gamma_acd_sqrt
    #Gamma_acd_2 = Kmatrix_acd.dot(Kmatrix_acd.T) / Kmatrix_acd.shape[1]
    #Gamma_acd_sqrt_2 = scipy.linalg.sqrtm(Gamma_acd_2)
    # Gamma_acd = np.linalg.pinv(Kmatrix_acd.T.dot(Kmatrix_acd) * Kmatrix_acd.shape[1], hermitian=True)

    Fmatrix = Kmatrix_acd.dot(Gamma_acd).dot(Kmatrix_acd.T) / (n_class * (n_class-1) / 2) / n_domain
    #Fmatrix = Gamma_acd_sqrt_2 @ Fmatrix @ Gamma_acd_sqrt_2
    Fmatrix = Fmatrix.real

    Kmatrix_add = np.zeros((n, n_class * n_domain))
    for iclass_idx in range(n_class):
        iclass = yclasslist[iclass_idx]
        for idomain_idx in range(n_domain):
            idomain = xclasslist[idomain_idx]
            idxi = (Xlabel == idomain)
            idxj = (Y[idxi] == iclass)
            Kmatrix_add[:, iclass_idx*n_domain + idomain_idx] = Kmatrix[:, idxi][:, idxj].mean(axis=1)
    # Gamma_add = Kmatrix_add[:, 0:n_domain].T.dot(Kmatrix_add[:, 0:n_domain])
    # for iclass_idx in range(1, n_class):
    #     temp_mat = Kmatrix_add[:, (iclass_idx*n_domain):((iclass_idx+1)*n_domain)]
    #     Gamma_add = scipy.linalg.block_diag(Gamma_add, temp_mat.T.dot(temp_mat))
    # Gamma_add = np.linalg.pinv(Gamma_add * Kmatrix_add.shape[1], hermitian=True)

    Gamma_add = np.zeros((n_class * n_domain, n_class * n_domain))
    for iclass_idx in range(n_class):
        iclass = yclasslist[iclass_idx]
        for idomain_idx in range(n_domain):
            idomain = xclasslist[idomain_idx]
            idxi = (Xlabel == idomain)
            idxj = (Y[idxi] == iclass)
            Gamma_add = Gamma_add + np.cov(Kmatrix_add[idxi, :][idxj, :].T)
    Gamma_add = np.linalg.pinv(Gamma_add / n_class / n_domain, hermitian=True)

    #Gamma_add = np.linalg.pinv(np.cov(Kmatrix_add.T), hermitian=True)#Kmatrix_add.T.dot(Kmatrix_add) #/ Kmatrix_add.shape[0]
    Gamma_add_sqrt = scipy.linalg.sqrtm(Gamma_add)

    for iclass_idx in range(n_class):
        iclass = yclasslist[iclass_idx]
        for idomain_idx in range(n_domain):
            idomain = xclasslist[idomain_idx]
            idxi = (Xlabel == idomain)
            idxj = (Y[idxi] == iclass)
            Kmatrix_add_block = Kmatrix_add[idxi, :][idxj, :]
            Kmatrix_mean = Kmatrix_add_block.mean(axis=0)
            Kmatrix_add_block = Kmatrix_add_block - Kmatrix_mean
            Kmatrix_add_block = Kmatrix_add_block @ Gamma_add_sqrt
            Kmatrix_add_block = Kmatrix_add_block + Kmatrix_mean
            Kmatrix_add[idxi, :][idxj, :] = Kmatrix_add_block

    Gamma_add = n_domain * np.eye(n_domain) - np.outer(np.ones(n_domain), np.ones(n_domain))
    Gamma_add = np.kron(np.eye(n_class, dtype=int), Gamma_add)
    #Gamma_add = Gamma_add_sqrt @ Gamma_add @ Gamma_add_sqrt
    #Gamma_add_2 = Kmatrix_add.dot(Kmatrix_add.T) / Kmatrix_add.shape[1]
    #Gamma_add_sqrt_2 = scipy.linalg.sqrtm(Gamma_add_2)
    # Gamma_add = np.linalg.pinv(Kmatrix_add.T.dot(Kmatrix_add) * Kmatrix_add.shape[1], hermitian=True)

    Gmatrix = Kmatrix_add.dot(Gamma_add).dot(Kmatrix_add.T) / (n_domain * (n_domain-1) / 2) / n_class
    #Gmatrix = Gamma_add_sqrt_2 @ Gmatrix @ Gamma_add_sqrt_2
    Gmatrix = Gmatrix.real

    Cmatrix = Fmatrix
    Qmatrix = Gmatrix + gamma * Kmatrix + epsilon * np.eye(n)

    w, Bmatrix = scipy.linalg.eigh(Cmatrix, Qmatrix, subset_by_index=[n-qmax, n-1])
    w = w.real
    Bmatrix = Bmatrix.real
    B_norm = np.diag(Bmatrix.T @ Kmatrix @ Bmatrix)

    # B_norm = (Bmatrix ** 2).sum(axis=0)
    # w = w / B_norm
    Bmatrix = Bmatrix / (np.abs(B_norm[None, :]) ** (1 / 2))
    # ind = np.argpartition(np.abs(w), qmax)[-qmax:]
    # Bmatrix = Bmatrix[:, ind]
    # Bmatrix = Bmatrix / (np.abs(w[None, ind])**(1/2))
    # Bmatrix = Bmatrix / (np.abs(w[None, :]) ** (1 / 2))
    return Bmatrix, Kmatrix, w
