import numpy as np
from scipy import linalg as LA
from numba import njit

#------------------------------------------------------------------------------------------------------------------------
#                                   USEFUL FUNCTIONS TO ANALYZE THE GENERATED DATA
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#    HAMMING DISTANCES BETWEEN MSAs
#------------------------------------------------------------------------------------------------------------------------


# Hamming distance between 2 sequences
@njit()
def H_distance(seq, seq1):
    '''
    seq,    the 1d arrays of tokens of the 2 sequences, they should not have the
    seq1:   first token (0) and they should end before the start of the padding tokens (1).
    '''
    return np.sum(((seq - seq1) != 0) + 1 - 1) / len(seq)


# Hamming distance between all sequences in an MSA (unordered array of values)
@njit()
def H_histo(tkn):
    '''
    tkn:    the 2d array of tokens of one MSA, it should not have the first token (0)
            and it should end before the start of the padding tokens (1).
    '''
    H_list = np.zeros((int(tkn.shape[0] * (tkn.shape[0] - 1) / 2)))
    k = 0
    for i in range(tkn.shape[0]):
        for j in range(i + 1, tkn.shape[0]):
            H_list[k] = H_distance(tkn[i, :], tkn[j, :])
            k += 1
    return H_list


# Hamming distance between all sequences in an MSA (matrix of all distances)
@njit()
def H_matrix(tkn):
    '''
    tkn:    the 2d array of tokens of one MSA, it should not have the first token (0)
            and it should end before the start of the padding tokens (1).
    '''
    H_mat = np.zeros((tkn.shape[0], tkn.shape[0]))

    for i in range(tkn.shape[0]):
        for j in range(i, tkn.shape[0]):
            H_mat[i, j] = H_distance(tkn[i, :], tkn[j, :])

    return H_mat + H_mat.T


# Hamming distances of all sequences in an MSA w.r.t. the nearest sequence in the
# original MSA (unordered array of values)
@njit()
def Cross_H_dist(old_tkn, new_tkn):
    '''
    old_tkn,    the 2d arrays of tokens of one MSA (their depth can be different but
    new_tkn:    not the length of the sequences), they shouldn't have the first token
                (0) and they should end before the start of the padding tokens (1).
    '''
    H_dist = np.zeros(new_tkn.shape[0]) + 1
    for i in range(
            new_tkn.shape[0]
    ):  #only i can be parallelized otherwise cross-iteration dependencies
        for j in range(old_tkn.shape[0]):
            dd = H_distance(old_tkn[j, :], new_tkn[i, :])
            if dd <= H_dist[i]:
                H_dist[i] = dd
    return H_dist


#------------------------------------------------------------------------------------------------------------------------
#   TOKENS/AMINO-ACIDS STATISTICS IN THE MSA
#------------------------------------------------------------------------------------------------------------------------


# Computes the 3-pt frequencies f_{i,j,k}(A,B,C) (and the connected correlations c_{i,j,k}(A,B,C)) of all the amino acids (different tokens)
# in 2 MSAs. The result is a 2d histogram which gives the number of frequencies/correlations which assume the values of the tuple (x,y)
# where x is f/c_{i,j,k}(A,B,C) of MSA1 and y is f/c_{i,j,k}(A,B,C) of MSA2.
@njit()
def Stat_3pt(seqs1, seqs2, val=0.5, bins=500):
    '''
    seqs1,      the 2d arrays of tokens of 2 MSAs of shape: (depth msa, len seq) which we want to compare, they shouldn't
    seqs2:      have the first token (0) and they should end before the start of the padding tokens (1).
    val:        the maximal absolute value for the correlations, the histogram will have points in (-val,val)x(-val,val).
    bins:       number of bins in 1 dimension of the histogram, the 2d one will have (bins)x(bins) bins.
    '''
    assert seqs1.shape == seqs2.shape
    depth = seqs1.shape[0]
    length = seqs1.shape[1]
    bin_width = (val * 2) / bins
    bin_width_f = 1 / bins

    N_tks = int(np.max(seqs1) + 1)
    len = int(bins) + 1
    hist = np.zeros((len, len), dtype=np.uint64)
    hist_f = np.zeros((len, len), dtype=np.uint64)
    lim = (len - 1) * bin_width / 2
    for i in range(length):
        for j in range(i):
            for k in range(j):
                results_f = np.empty((2, N_tks**3), dtype=np.float64)
                results = np.empty((2, N_tks**3), dtype=np.float64)
                for a, seqs in enumerate([seqs1, seqs2]):
                    c_ijk = np.zeros((N_tks, N_tks, N_tks), dtype=np.float64)
                    for seq in seqs:
                        c_ijk[seq[i], seq[j], seq[k]] += 1

                    temp = c_ijk.sum(axis=0)
                    f_jk = temp.reshape((1, temp.shape[0], temp.shape[1]))
                    f_j = temp.sum(axis=1).reshape((1, temp.shape[0], 1))
                    f_k = temp.sum(axis=0).reshape((1, 1, temp.shape[1]))
                    temp = c_ijk.sum(axis=1)
                    f_ik = temp.reshape((temp.shape[0], 1, temp.shape[1]))
                    f_i = temp.sum(axis=1).reshape((temp.shape[0], 1, 1))
                    temp = c_ijk.sum(axis=2)
                    f_ij = temp.reshape((temp.shape[0], temp.shape[1], 1))

                    c_ijk /= depth
                    results_f[a, :] = c_ijk.flatten()

                    c_ijk -= (f_ij * f_k + f_ik * f_j + f_jk * f_i) / (depth**
                                                                       2)
                    c_ijk += (2 * f_i * f_j * f_k) / (depth**3)
                    results[a, :] = c_ijk.flatten()

                bin_idxs = ((results + lim) / bin_width).astype(
                    np.int64).flatten()
                bin_idxs[bin_idxs < 0] = 0
                bin_idxs[bin_idxs > len - 1] = len - 1
                bin_idxs = bin_idxs.reshape((2, -1))

                bin_idxs1 = ((results_f) / bin_width_f).astype(
                    np.int64).flatten()
                bin_idxs1[bin_idxs1 < 0] = 0
                bin_idxs1[bin_idxs1 > len - 1] = len - 1
                bin_idxs1 = bin_idxs1.reshape((2, -1))

                for p in range(bin_idxs.shape[1]):
                    row_idx = bin_idxs[0, p]
                    col_idx = bin_idxs[1, p]
                    hist[row_idx, col_idx] += 1

                    row_idx1 = bin_idxs1[0, p]
                    col_idx1 = bin_idxs1[1, p]
                    hist_f[row_idx1, col_idx1] += 1
    return hist, hist_f


# Computes the 2-pt frequencies f_{i,j}(A,B) (and the connected correlations c_{i,j}(A,B)) of all the amino acids (different tokens)
# in 2 MSAs. The result is a 2d histogram which gives the number of frequencies/correlations which assume the values of the tuple (x,y)
# where x is f/c_{i,j}(A,B) of MSA1 and y is f/c_{i,j}(A,B) of MSA2.
@njit()
def Stat_2pt(seqs1, seqs2, val=1, bins=100):
    '''
    seqs1,      the 2d arrays of tokens of 2 MSAs of shape: (depth msa, len seq) which we want to compare, they shouldn't
    seqs2:      have the first token (0) and they should end before the start of the padding tokens (1).
    val:        the maximal absolute value for the correlations, the histogram will have points in (-val,val)x(-val,val).
    bins:       number of bins in 1 dimension of the histogram, the 2d one will have (bins)x(bins) bins.
    '''
    assert seqs1.shape == seqs2.shape
    depth = seqs1.shape[0]
    length = seqs1.shape[1]
    bin_width = (val) / bins
    bin_width_f = 1 / bins

    N_tks = int(np.max(seqs1) + 1)
    len = int(bins) + 1
    hist = np.zeros((len, len), dtype=np.uint64)
    hist_f = np.zeros((len, len), dtype=np.uint64)
    lim = (len - 1) * bin_width / 2
    for i in range(length):
        for j in range(i):
            results_f = np.empty((2, N_tks**2), dtype=np.float64)
            results = np.empty((2, N_tks**2), dtype=np.float64)
            for a, seqs in enumerate([seqs1, seqs2]):
                c_ij = np.zeros((N_tks, N_tks), dtype=np.float64)
                for seq in seqs:
                    c_ij[seq[i], seq[j]] += 1

                temp = c_ij.sum(axis=0)
                f_j = temp.reshape((1, temp.shape[0]))
                temp = c_ij.sum(axis=1)
                f_i = temp.reshape((temp.shape[0], 1))

                c_ij /= depth
                results_f[a, :] = c_ij.flatten()

                c_ij -= (f_j * f_i) / (depth**2)
                results[a, :] = c_ij.flatten()

            bin_idxs = ((results + lim) / bin_width).astype(np.int64).flatten()
            bin_idxs[bin_idxs < 0] = 0
            bin_idxs[bin_idxs > len - 1] = len - 1
            bin_idxs = bin_idxs.reshape((2, -1))

            bin_idxs1 = ((results_f) / bin_width_f).astype(np.int64).flatten()
            bin_idxs1[bin_idxs1 < 0] = 0
            bin_idxs1[bin_idxs1 > len - 1] = len - 1
            bin_idxs1 = bin_idxs1.reshape((2, -1))

            for p in range(bin_idxs.shape[1]):
                row_idx = bin_idxs[0, p]
                col_idx = bin_idxs[1, p]
                hist[row_idx, col_idx] += 1

                row_idx1 = bin_idxs1[0, p]
                col_idx1 = bin_idxs1[1, p]
                hist_f[row_idx1, col_idx1] += 1
    return hist, hist_f


# Computes the 1-pt frequencies f_{i}(A) of all the amino acids (different tokens)
# in 2 MSAs. The result is a 2d histogram which gives the number of frequencies
# which assume the values of the tuple (x,y) where x is f_{i}(A) of MSA1 and y is f_{i}(A) of MSA2.
@njit()
def Stat_1pt(seqs1, seqs2, val=1, bins=100):
    '''
    seqs1,      the 2d arrays of tokens of 2 MSAs of shape: (depth msa, len seq) which we want to compare, they shouldn't
    seqs2:      have the first token (0) and they should end before the start of the padding tokens (1).
    val:        the maximal absolute value for the correlations, the histogram will have points in (-val,val)x(-val,val)
    bins:       number of bins in 1 dimension of the histogram, the 2d one will have (bins)x(bins) bins
    '''
    assert seqs1.shape == seqs2.shape
    depth = seqs1.shape[0]
    length = seqs1.shape[1]
    bin_width = (val) / bins

    N_tks = int(np.max(seqs1) + 1)
    len = int(bins) + 1
    hist = np.zeros((len, len), dtype=np.uint64)
    for i in range(length):
        results = np.empty((2, N_tks), dtype=np.float64)
        for a, seqs in enumerate([seqs1, seqs2]):
            c_i = np.zeros((N_tks), dtype=np.float64)
            for seq in seqs:
                c_i[seq[i]] += 1

            c_i /= depth
            results[a, :] = c_i

        bin_idxs = ((results) / bin_width).astype(np.int64).flatten()
        bin_idxs[bin_idxs < 0] = 0
        bin_idxs[bin_idxs > len - 1] = len - 1
        bin_idxs = bin_idxs.reshape((2, -1))

        for p in range(bin_idxs.shape[1]):
            row_idx = bin_idxs[0, p]
            col_idx = bin_idxs[1, p]
            hist[row_idx, col_idx] += 1
    return hist


# Compute the probability distributions of the different amino acids for each column of the new and the original MSA.
# Then compute the joint distribution of the entries in the same column of the old and the new MSA
@njit()
def Cross_Probabilities(old_tkn, new_tkn, N_keys):
    '''
    old_tkn,    the 2d arrays of tokens of one MSA (their shape must be the same), they shouldn't
    new_tkn:    have the first token (0) and they should end before the start of the padding tokens (1).
    N_keys:     the number of different tokens that you can get in the MSA, i.e. # unique amino acids.
                The tokens must start from zero and get each possible integer until the max.
                Generally N_keys=len(token_dictionary.keys())
    '''
    N = old_tkn.shape[0]
    px = np.zeros((N_keys, old_tkn.shape[1]))
    py = np.zeros_like(px)
    pxy = np.zeros((N_keys, N_keys, old_tkn.shape[1]))
    for i in range(old_tkn.shape[1]):
        for j in range(N):
            px[old_tkn[j, i], i] += 1 / N
            py[new_tkn[j, i], i] += 1 / N
            pxy[old_tkn[j, i], new_tkn[j, i], i] += 1 / N
    return px, py, pxy


# Compute the single variable & joint Entropies and the Mutual information of the Probability distributions
# of the same columns of 2 different MSAs: P(x), P(y), P(x,y). The random variables x and y can assume a finite
# number of integer values given by the total number of unique tokens in the MSAs.
def Cross_MI_Ent(px, py, pxy):
    '''
    px,py:  the 2d arrays of probability distributions of the tokens for each column in the MSA, the shape of
            the arrays must be (# all unique amino acids, # columns in the MSA)
    pxy:    the 3d array of joint probability distributions of the tokens for the same column in the 2 MSAs,
            the shape of the array must be (# unique amino acids, # unique amino acids, # columns in the MSA)
    '''
    #Entropies
    Ex, Ey = -np.sum(px * np.log2(np.clip(px, 1e-12, None)), axis=0), -np.sum(
        py * np.log2(np.clip(py, 1e-12, None)), axis=0)
    Exy = -np.sum(pxy * np.log2(np.clip(pxy, 1e-12, None)), axis=(0, 1))
    #Mutual information
    Mutual = Ex + Ey - Exy
    '''
    # Another way of computing the Mutual information: SLOWER
    minfo = np.zeros((px.shape[1]))
    for i in range(px.shape[1]):
        M1 = px[:,i].reshape((px.shape[0], 1)) @ py[:,i].reshape((1, py.shape[0]))
        minfo[i] = np.sum(pxy[:,:,i]*(np.log2(np.clip(pxy[:,:,i], 1e-12, None))-np.log2(np.clip(M1, 1e-12, None))))
    '''
    return Mutual, Ex, Ey, Exy


# Compute all the Entropies, the Mutual information and the Coinformation independently for 2 different MSAs
# Then populates a 2d histogram which gives the number of elements which assume the values of the tuple (x,y)
# where x are values for the first MSA and y for the second.
@njit()
def Entropies_Information(seqs1,
                          seqs2,
                          val=np.array([10, 10, 10, 10, 10, 10, 10]),
                          bins=100):
    '''
    seqs1,      the 2d arrays of tokens of 2 MSAs of shape: (depth msa, len seq) which we want to compare, they shouldn't
    seqs2:      have the first token (0) and they should end before the start of the padding tokens (1).
    val:        the maximal absolute value for the outputs.
    bins:       number of bins in 1 dimension of the histogram, the 2d one will have (bins)x(bins) bins.
    '''
    assert seqs1.shape == seqs2.shape
    depth = seqs1.shape[0]
    length = seqs1.shape[1]
    bin_width = val / bins
    vars = val.shape[0]

    N_tks = int(np.max(seqs1) + 1)
    len = int(bins) + 1
    hist = np.zeros((vars, len, len), dtype=np.uint64)
    lim = np.zeros(7, dtype=np.float64)
    lim[-1], lim[-2] = float(bins * bin_width[-1] / 2), float(
        bins * bin_width[-2] / 2)
    # Corrections
    #C1, C2, C3 = N_tks/(2*depth), N_tks**2/(2*depth), N_tks**3/(2*depth)
    for i in range(length):
        for j in range(i):
            for k in range(j):
                results = np.empty((2, vars), dtype=np.float64)
                for a, seqs in enumerate([seqs1, seqs2]):
                    p_ijk = np.zeros((N_tks, N_tks, N_tks), dtype=np.float64)
                    for seq in seqs:
                        p_ijk[seq[i], seq[j], seq[k]] += 1
                    p_ijk /= depth
                    p_jk = p_ijk.sum(axis=0)
                    p_j = p_jk.sum(axis=1)
                    p_k = p_jk.sum(axis=0)
                    p_ik = p_ijk.sum(axis=1)
                    p_i = p_ik.sum(axis=1)
                    p_ij = p_ijk.sum(axis=2)
                    #Entropies
                    C1 = np.count_nonzero(p_i) / (2 * depth)
                    C2 = np.count_nonzero(p_ij) / (2 * depth)
                    C3 = np.count_nonzero(p_ijk) / (2 * depth)
                    results[a,
                            0] = -np.sum(p_i * np.log2(p_i + 10**(-15))) + C1
                    E_j = -np.sum(p_j * np.log2(p_j + 10**(-15))) + C1
                    E_k = -np.sum(p_k * np.log2(p_k + 10**(-15))) + C1
                    results[a,
                            1] = -np.sum(p_ij * np.log2(p_ij + 10**(-15))) + C2
                    E_ik = -np.sum(p_ik * np.log2(p_ik + 10**(-15))) + C2
                    E_jk = -np.sum(p_jk * np.log2(p_jk + 10**(-15))) + C2
                    results[a, 2] = -np.sum(p_ijk * np.log2(p_ijk + 10**
                                                            (-15))) + C3
                    #Mutual information
                    results[a, 3] = results[a, 0] + E_j - results[a, 1]
                    #Score Mutual
                    results[a, 4] = results[a, 3] / results[a, 1]
                    #Coinformation
                    results[a, 5] = results[a, 0] + E_j + E_k - (
                        results[a, 1] + E_ik + E_jk) + results[a, 2]
                    #Score Coinfo
                    results[a, 6] = results[a, 5] / results[a, 2]

                for id in range(vars):
                    bin_idxs = ((results[:, id] + lim[id]) /
                                bin_width[id]).astype(np.int64)
                    bin_idxs[bin_idxs < 0] = 0
                    bin_idxs[bin_idxs > len - 1] = len - 1
                    bin_idxs = bin_idxs  #.reshape((2,))
                    hist[id, bin_idxs[0], bin_idxs[1]] += 1
    '''
    Output: 2d-histograms, each one for the quantities:
            H(X), H(X,Y), H(X,Y,Z), I(X;Y), Score(X;Y), I(X;Y;Z), Score(X;Y;Z)
    '''
    return hist


# Compute all the Pointwise Mutual informations and Coinformations independently for 2 different MSAs
# Then populates a 2d histogram which gives the number of elements which assume the values of the tuple (x,y)
# where x are values for the first MSA and y for the second.
@njit()
def Pointwise_Infos(seqs1, seqs2, val=20, bins=1000):
    '''
    seqs1,      the 2d arrays of tokens of 2 MSAs of shape: (depth msa, len seq) which we want to compare, they shouldn't
    seqs2:      have the first token (0) and they should end before the start of the padding tokens (1).
    val:        the maximal absolute value for the outputs.
    bins:       number of bins in 1 dimension of the histogram, the 2d one will have (bins)x(bins) bins.
    '''
    assert seqs1.shape == seqs2.shape
    depth = seqs1.shape[0]
    length = seqs1.shape[1]
    bin_width = val / bins

    N_tks = int(np.max(seqs1) + 1)
    len = int(bins) + 1
    lim = val / 2
    hist = np.zeros((2, len, len), dtype=np.uint64)
    for i in range(length):
        for j in range(i):
            for k in range(j):
                results = np.empty((2, N_tks**2), dtype=np.float64)
                results2 = np.empty((2, N_tks**3), dtype=np.float64)
                for a, seqs in enumerate([seqs1, seqs2]):
                    p_ijk = np.zeros((N_tks, N_tks, N_tks), dtype=np.float64)
                    for seq in seqs:
                        p_ijk[seq[i], seq[j], seq[k]] += 1
                    p_ijk /= depth
                    temp = p_ijk.sum(axis=0)
                    p_jk = temp.reshape((1, temp.shape[0], temp.shape[1]))
                    p_j = temp.sum(axis=1).reshape((1, temp.shape[0], 1))
                    p_k = temp.sum(axis=0).reshape((1, 1, temp.shape[1]))
                    temp = p_ijk.sum(axis=1)
                    p_ik = temp.reshape((temp.shape[0], 1, temp.shape[1]))
                    p_i = temp.sum(axis=1).reshape((temp.shape[0], 1, 1))
                    temp = p_ijk.sum(axis=2)
                    p_ij = temp.reshape((temp.shape[0], temp.shape[1], 1))

                    # Compute pointwise MI and Coinfo
                    vv = np.zeros_like(p_ijk[:, :, 0]).flatten()
                    idxs = (p_ij[:, :, 0] * p_i[:, :, 0] *
                            p_j[:, :, 0]).flatten()
                    vv[idxs > 0] = 1
                    vv = vv.reshape(p_ijk[:, :, 0].shape)
                    results[a, :] = (vv * np.log2(
                        (p_ij[:, :, 0] + 10**(-20)) /
                        (p_i[:, :, 0] * p_j[:, :, 0] + 10**(-20)))).flatten()
                    vv = np.zeros_like(p_ijk).flatten()
                    idxs = (p_ij * p_ik * p_jk * p_i * p_j * p_k *
                            p_ijk).flatten()
                    vv[idxs > 0] = 1
                    vv = vv.reshape(p_ijk.shape)
                    results2[a, :] = (vv * np.log2(
                        (p_ij * p_ik * p_jk + 10**(-20)) /
                        (p_i * p_j * p_k * p_ijk + 10**(-20)))).flatten()

                bin_idxs = ((results + lim) / bin_width).astype(
                    np.int64).flatten()
                bin_idxs[bin_idxs < 0] = 0
                bin_idxs[bin_idxs > len - 1] = len - 1
                bin_idxs = bin_idxs.reshape((2, -1))

                for p in range(bin_idxs.shape[1]):
                    row_idx = bin_idxs[0, p]
                    col_idx = bin_idxs[1, p]
                    hist[0, row_idx, col_idx] += 1

                bin_idxs = ((results2 + lim) / bin_width).astype(
                    np.int64).flatten()
                bin_idxs[bin_idxs < 0] = 0
                bin_idxs[bin_idxs > len - 1] = len - 1
                bin_idxs = bin_idxs.reshape((2, -1))

                for p in range(bin_idxs.shape[1]):
                    row_idx = bin_idxs[0, p]
                    col_idx = bin_idxs[1, p]
                    hist[1, row_idx, col_idx] += 1
    '''
    Output: 2d-histograms, each one for the quantities:
            P.M.I.(X,Y), P.C.I(X;Y;Z)
    '''
    return hist


# Compute all the Frobenius norms for the 2 & 3 point correlation matrices at fixed i,j,k independently for 2 different MSAs
# Then populates a 2d histogram which gives the number of elements which assume the values of the tuple (x,y)
# where x are values for the first MSA and y for the second.
@njit()
def Frobenius(seqs1, seqs2, val=np.array([2,2]), bins=100):
    '''
    seqs1,      the 2d arrays of tokens of 2 MSAs of shape: (depth msa, len seq) which we want to compare, they shouldn't
    seqs2:      have the first token (0) and they should end before the start of the padding tokens (1).
    val:        the maximal absolute value for the outputs.
    bins:       number of bins in 1 dimension of the histogram, the 2d one will have (bins)x(bins) bins.
    '''
    assert seqs1.shape == seqs2.shape
    depth = seqs1.shape[0]
    length = seqs1.shape[1]
    bin_width = val / bins
    vars = val.shape[0]

    N_tks = int(np.max(seqs1) + 1)
    len = int(bins) + 1
    hist = np.zeros((vars, len, len), dtype=np.uint64)
    for i in range(length):
        for j in range(i):
            for k in range(j):
                results = np.empty((2,vars), dtype=np.float64)
                for a, seqs in enumerate([seqs1, seqs2]):
                    p_ijk = np.zeros((N_tks, N_tks, N_tks), dtype=np.float64)
                    for seq in seqs:
                        p_ijk[seq[i], seq[j], seq[k]] += 1
                    p_ijk /= depth
                    temp = p_ijk.sum(axis=0)
                    p_jk = temp.reshape((1, temp.shape[0], temp.shape[1]))
                    p_j = temp.sum(axis=1).reshape((1, temp.shape[0], 1))
                    p_k = temp.sum(axis=0).reshape((1, 1, temp.shape[1]))
                    temp = p_ijk.sum(axis=1)
                    p_ik = temp.reshape((temp.shape[0], 1, temp.shape[1]))
                    p_i = temp.sum(axis=1).reshape((temp.shape[0], 1, 1))
                    temp = p_ijk.sum(axis=2)
                    p_ij = temp.reshape((temp.shape[0], temp.shape[1], 1))

                    P2 = p_ij[:,:,0] - p_i[:,:,0]*p_j[:,:,0]
                    results[a,0] = np.sqrt(np.sum(np.square(P2)))
                    P3 = p_ijk - (p_ij * p_k + p_ik * p_j + p_jk * p_i) + 2 * p_i * p_j * p_k
                    results[a,1] = np.sqrt(np.sum(np.square(P3)))

                for id in range(vars):
                    bin_idxs = ((results[:,id]) / bin_width[id]).astype(np.int64)
                    bin_idxs[bin_idxs < 0] = 0
                    bin_idxs[bin_idxs > len - 1] = len - 1

                    hist[id, bin_idxs[0], bin_idxs[1]] += 1

    return hist


# Compares the Generated MSA to the original one by summing at each residue in each sequence
# the weight associated to the amino acid couple found in the same residue of the new and the
# original MSA. The weights are given by a substitution matrix (generally blosum62)
def Compare_MSA(old_tkn, new_tkn, blosum, dic_inv):
    '''
    old_tkn,    the 2d arrays of tokens of one MSA (their shape must be the same), they shouldn't
    new_tkn:    have the first token (0) and they should end before the start of the padding tokens (1).
    blosum:     is a pandas dataframe which contains all the weights of the substitution matrix, its headers
                must be the strings of all possible amino acids (generally they are 20 + gap + 3 uncommon ones).
    dic_inv:    the inverted dictionary of all the tokens in old_tkn & new_tkn, for each token
                value int() it assigns the associated amino acid character str().
    '''
    score = 0
    for idx in np.ndindex(old_tkn.shape):
        a = dic_inv[old_tkn[idx]]
        b = dic_inv[new_tkn[idx]]
        score += blosum[a][b]
    return score


#------------------------------------------------------------------------------------------------------------------------
#   OTHERS
#------------------------------------------------------------------------------------------------------------------------


# Convert the tokens of an MSA associated to the dictionary dic1 into the same array with new
# token labels given from the desired labels in dic2
def Convert_tokens(dic1, tkn1, dic2):
    """
    tkn1 can have any shape, generally it's just an MSA.
    dic1:   token dictionary of tkn1
    dic2:   token dictionary of tkn2 (the desired tokens)
    """
    dic1_inv = dict(zip(dic1.values(), dic1.keys()))
    #dic2_inv = dict(zip(dic2.values(), dic2.keys()))
    tkn2 = np.zeros_like(tkn1)
    for idx, x in np.ndenumerate(tkn1):
        letter = dic1_inv[int(x)]
        tkn2[idx] = dic2[letter]
    return tkn2


# Sets to zero the elements in all the diagonals from -n to +n of a square matrix
# It's used to modify the contact matrices by eliminating the central diagonals which
# have high values only because the residues are neighbors
def Zero_diag(mat, n):
    '''
    mat:    square matrix (generally it's used for the contact matrix)
    n:      is the  number of diagonals that you want to cancel above and below the main diagonal
    '''
    def diag_idx(a, k):
        tmp = np.diag(np.ones(int(a.shape[-1] - abs(k))), k=k)
        idx = tmp == 1
        return idx

    b = np.copy(mat)
    for i in range(-n, n + 1):
        b[..., diag_idx(b, i)] = 0
    return b


# Does the PCA of all the sequences in an MSA and return their projection into the first 10 principal components
def PCA(x, y=None):
    '''
    If there is only the input x (2d array of tokens of one MSA without first (0) and padding (1) tokens)
    the function return its PCA.
    If there are 2 input MSAs x,y the function return the projection of both of them into the direction
    given by the principal components of the FIRST MSA.
    '''
    x -= np.mean(x, axis=0)
    #x = x/(np.std(x, axis = 0)+10**(-5))
    cov = np.cov(x, rowvar=False)
    evals, evecs = LA.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    #evals = evals[idx]
    if y is not None:
        y -= np.mean(y, axis=0)
        #y = y/(np.std(y, axis = 0)+10**(-5))
        return np.dot(x, evecs[:, :10]), np.dot(y, evecs[:, :10])
    else:
        return np.dot(x, evecs[:, :10])


# Transform the original MSA into an MSA of concatenated one-hot encodings of each residue, the new MSA
# has the shape (depth msa, (len seq)*#amino acids)
# The output MSA then can be given as input of the PCA() function, the results sometimes are different
def One_hot(tkn, N_keys):
    '''
    tkn:        the 2d array of tokens of one MSA (without first (0) and padding (1) tokens)
    N_keys:     the number of different tokens that you can get in the MSA, i.e. # unique amino acids.
                The tokens must start from zero and get each possible integer until the max.
                Generally N_keys=len(token_dictionary.keys())
    '''
    a = np.zeros((tkn.shape[0], tkn.shape[1] * N_keys))
    for k in range(tkn.shape[0]):
        b = np.zeros((tkn.shape[1], N_keys))
        b[np.arange(tkn.shape[1]), tkn[k, :]] = 1
        a[k, :] = b.ravel()
    return a


#------------------------------------------------------------------------------------------------------------------------
#   DEPRECATED FUNCTIONS FOR STATISTICS OF MSAs
#------------------------------------------------------------------------------------------------------------------------


# 1-pt & 2-pt relative frequencies f_{i,j}(A,B) of all the amino acids (different tokens)
# in an MSA. The result is a 4d array of shape (#amino acids, #amino acids, len seq, len seq)
@njit()
def Frequencies(tkn):
    '''
    tkn:    the 2d array of tokens of one MSA of shape: (depth msa, len seq), they shouldn't
            have the first token (0) and they should end before the start of the padding tokens (1).
    The fastest results are obtained when the tokens start from zero (gap) and reach a
    maximum integer value (generally 21 if there are no strange amino acids).
    '''
    N_tks = int(np.max(tkn) + 1)
    freqs = np.zeros((N_tks, N_tks, tkn.shape[1], tkn.shape[1]))
    for i in range(tkn.shape[1]):
        for j in range(i, tkn.shape[1]):
            for k in range(tkn.shape[0]):
                ind_i, ind_j = tkn[k, i], tkn[k, j]
                if i != j:
                    freqs[ind_i, ind_j, i, j] += 1
                    freqs[ind_j, ind_i, j, i] += 1
                if i == j:
                    freqs[ind_i, ind_j, i, i] += 1
    rel_freqs = freqs / tkn.shape[0]
    # control normalization
    #tot = np.sum(rel_freqs, axis=(0,1))
    #if np.any(tot > 1 + 10**(-6)) or np.any(tot < 1 - 10**(-6)):
    #return 0
    #print('ERROR')
    #else:
    return rel_freqs


# Get the 1-pt relative frequencies from the frequency array computed by 'Frequencies()'
@njit()
def Get_1pt_freq(freqs):
    '''
    freqs:  array which should have the shape (#amino acids, #amino acids, len seq, len seq).
    '''
    a = np.zeros((freqs.shape[0], freqs.shape[2]))
    for i in range(freqs.shape[2]):
        a[:, i] = np.diag(freqs[:, :, i, i])
    return a


# Compute the Connected 2-pt relative frequencies in the MSA: f_{i,j}(A,B)-f_i(A)*f_j(B)
# If you compute the average over the last 2 axes of the result you get the correlations
# between the amino acids, an array of shape (#amino acids, #amino acids)
@njit()
def Get_correl(freqs):
    '''
    freqs:  array which should have the shape (#amino acids, #amino acids, len seq, len seq).
    '''
    ff = Get_1pt_freq(freqs)
    corr = np.zeros_like(freqs)
    for i in range(freqs.shape[2]):
        for j in range(i, freqs.shape[3]):
            diags = ff[:, i].copy().reshape(
                (ff.shape[0], 1)) @ ff[:, j].copy().reshape((1, ff.shape[0]))
            if i == j:
                corr[:, :, i, i] = freqs[:, :, i, i] - diags
            else:
                corr[:, :, i, j] = freqs[:, :, i, j] - diags
                corr[:, :, j, i] = freqs[:, :, j, i] - diags.T
    return corr