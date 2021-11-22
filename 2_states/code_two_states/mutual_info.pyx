# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

import numpy as np
from libc.math cimport log
from scipy.optimize import linear_sum_assignment
from cython.parallel import prange
import time


def Inference_Partner_Mutual_Info(l_msa, s_train, reg, n_pair, fast=False, middle_index=None, n_mean=30, **kwargs):
    cdef :
        int i_avg, index_test, ind_species, counter_average = 0
        double[::1] Percentage_True_partner = np.zeros(n_pair)
        
    if middle_index is None:
        middle_index = int(float(l_msa[0].shape[1])/2.0)

    for i_avg in range(n_mean):
        for index_test in range(l_msa.shape[0]):
            msa = np.array(((l_msa[index_test]+1)/2), dtype=np.int8)
            l_perm = np.random.permutation(l_msa.shape[1])
            msa_train = msa[l_perm[:s_train]]
            if fast:
                index_max = min(s_train + 50*n_pair,np.size(l_msa, axis=1))
                msa_test = msa[l_perm[s_train:index_max]]
            else:
                msa_test = msa[l_perm[s_train:]]
                
            PMI_matrix = PMI_cython(msa_train, reg, middle_index)
            ind_species = n_pair
            
            while ind_species<msa_test.shape[0]:
                Cost = np.zeros((n_pair,n_pair))
                Score_Sij_cython(Cost, msa_test[ind_species-n_pair:ind_species], PMI_matrix, middle_index)
                Permutation_worker_row = np.random.permutation(Cost.shape[0])
                Cost = Cost[Permutation_worker_row]
                row_ind, col_ind = linear_sum_assignment(Cost)
                Percentage_True_partner += Permutation_worker_row[row_ind] == col_ind # The true index of worker i is Permutation_worker[i]
                counter_average += 1
                ind_species += n_pair
        
    return np.mean(Percentage_True_partner)/counter_average


cdef void Score_Sij_cython(double[:,::1] S_AB, char[:,::1] msa_test, double[:,:,:,::1] PMI, int middle_index):
    cdef:
        int ind_liste, ind_liste_2, i, j
        
    for ind_liste in prange(S_AB.shape[0], nogil=True):
        for ind_liste_2 in range(S_AB.shape[1]):
            for i in range(middle_index):
                for j in range(middle_index, msa_test.shape[1]):
                    S_AB[ind_liste,ind_liste_2] -= PMI[i,j,msa_test[ind_liste,i],msa_test[ind_liste_2,j]]
                    

cpdef PMI_cython(char[:,::1] msa, double pseudocount, middle_index_prot=None):
    cdef :
        int site_i,site_j,index_spin_1,index_spin_2,index_try
        int middle_index
        float compteur = 0
        ##PMI[site_i,site_j,spin_i,spin_j]
        double[:,:,:,::1] PMI = np.zeros((msa.shape[1],msa.shape[1], 2, 2))
        double[:,:,:,::1] Fij = np.zeros((msa.shape[1],msa.shape[1], 2, 2))
        double[:,::1] Frequence_1body = np.zeros((msa.shape[1], 2))
        
    if middle_index_prot is None:
        middle_index = int(float(msa.shape[1])/2.0)
    else:
        middle_index = int(middle_index_prot)
        
    two_body_freq_cython(Fij, msa, pseudocount, middle_index)
    
    for site_i in prange(msa.shape[1], nogil=True):
        for index_try in range(msa.shape[0]):
            Frequence_1body[site_i, msa[index_try, site_i]] += 1
                
        Frequence_1body[site_i,0] = pseudocount/2 + (1-pseudocount)*Frequence_1body[site_i,0]/msa.shape[0]
        Frequence_1body[site_i,1] = pseudocount/2 + (1-pseudocount)*Frequence_1body[site_i,1]/msa.shape[0]#1 - Frequence_1body[site_i,0]
        
    for site_i in range(middle_index):
        for site_j in range(middle_index,msa.shape[1]):
            for index_spin_1 in range(2):
                for index_spin_2 in range(2):
                    PMI[site_i,site_j,index_spin_1,index_spin_2] = log(Fij[site_i,site_j,index_spin_1,index_spin_2]/(Frequence_1body[site_i,index_spin_1]*Frequence_1body[site_j,index_spin_2]))
            
    return np.asarray(PMI)

cdef void two_body_freq_cython(double[:,:,:,::1] Fij, char[:,::1] msa, double pseudocount, int middle_index):
    cdef:
        int site_i,site_j,index_try,index_spin_1,index_spin_2
        double delta_freq =  1.0/(msa.shape[0])
        
    for site_i in prange(middle_index, nogil=True):
        for site_j in range(middle_index, msa.shape[1]):
            for index_try in range(msa.shape[0]): 
                Fij[site_i,site_j,msa[index_try, site_i],msa[index_try, site_j]] += delta_freq
                
            for index_spin_1 in range(2):
                for index_spin_2 in range(2):
                    Fij[site_i,site_j,index_spin_1,index_spin_2] = pseudocount/4 + (1-pseudocount)*Fij[site_i,site_j,index_spin_1,index_spin_2]
             