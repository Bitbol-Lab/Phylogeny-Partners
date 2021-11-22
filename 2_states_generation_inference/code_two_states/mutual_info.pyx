# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

import numpy as np
cimport cython
from libc.math cimport log
from scipy.optimize import linear_sum_assignment


def Inference_Partner_Mutual_Info(l_msa, s_train, reg, n_pair, theta=0, fast=False, middle_index=None):
    n_mean = 20
    Percentage_True_partner = np.zeros(n_pair)
    counter_average = 0
    
    for index_test in range(l_msa.shape[0]):
        l_perm = np.random.permutation(l_msa.shape[1])
        msa_train = l_msa[index_test,l_perm[:s_train]]
        if fast :
            index_max = min(s_train + 50*n_pair,np.size(l_msa,axis=1))
            msa_test = l_msa[index_test,l_perm[s_train:index_max]]
        else :
            msa_test = l_msa[index_test, l_perm[s_train:]]
            
        PMI_matrix = PMI_cython(msa_train, reg, middle_index)
        ind_species = n_pair
        Cost = np.zeros((n_pair,n_pair))
        
        while ind_species<msa_test.shape[0]:
            Cost = -1*Score_Sij_cython(Cost, msa_test[ind_species-n_pair:ind_species], PMI_matrix, middle_index)
            Permutation_worker_row = np.random.permutation(Cost.shape[0])
            Cost = Cost[Permutation_worker_row]
            row_ind, col_ind = linear_sum_assignment(Cost)
            Percentage_True_partner += Permutation_worker_row[row_ind] == col_ind # The true index of worker i is Permutation_worker[i]
            counter_average += 1
            ind_species+=n_pair
            
    return np.mean(Percentage_True_partner/counter_average)


def Score_Sij_cython(double[:,::1] S_AB, char[:,::1] msa_test, double[:,:,:,::1] PMI, middle_index_prot = None):
    cdef:
        int middle_index 
        int ind_liste,ind_liste_2,i,j
        
    if middle_index_prot is None:
        middle_index = int(float(msa_test.shape[1])/2.0)
    else:
        middle_index = int(middle_index_prot)
        
    for ind_liste in range(S_AB.shape[0]):
        for ind_liste_2 in range(S_AB.shape[1]):
            S_AB[ind_liste,ind_liste_2] = 0
            for i in range(middle_index):
                for j in range(middle_index,msa_test.shape[1]):
                    S_AB[ind_liste,ind_liste_2] += PMI[i,j,int((msa_test[ind_liste,i]+1)/2),int((msa_test[ind_liste_2,j]+1)/2)]
                    
    return np.asarray(S_AB)


cpdef void two_body_freq_cython(double[:,:,:,::1] Fij, char[:,::1] MSA, double pseudocount, int middle_index):
    cdef:
        int site_i,site_j,index_try,index_spin_1,index_spin_2
        double delta_freq =  1.0/(MSA.shape[0])
        
    for site_i in range(middle_index):
        for site_j in range(middle_index,MSA.shape[1]):
            for index_try in range(MSA.shape[0]): 
                Fij[site_i,site_j,int((MSA[index_try,site_i]+1)/2.0),int((MSA[index_try,site_j]+1)/2.0)] += delta_freq
                
            for index_spin_1 in range(2):
                for index_spin_2 in range(2):
                    Fij[site_i,site_j,index_spin_1,index_spin_2] = pseudocount/4 + (1-pseudocount)*Fij[site_i,site_j,index_spin_1,index_spin_2]
            
            
def PMI_cython(char[:,::1] MSA, double pseudocount, middle_index_prot = None ):
    cdef :
        int site_i,site_j,index_spin_1,index_spin_2,index_try
        int middle_index
        float compteur = 0
        ##PMI[site_i,site_j,spin_i,spin_j]
        double[:,:,:,::1] PMI = np.zeros((MSA.shape[1],MSA.shape[1],2,2))
        double[:,:,:,::1] Fij = np.zeros((MSA.shape[1],MSA.shape[1],2,2))
        double[:,::1] Frequence_1body = np.zeros((MSA.shape[1],2))
        
    if middle_index_prot is None:
        middle_index = int(float(MSA.shape[1])/2.0)
    else:
        middle_index = int(middle_index_prot)
        
    two_body_freq_cython(Fij, MSA, pseudocount, middle_index)
    
    for site_i in range(MSA.shape[1]):
        compteur = 0
        for index_try in range(MSA.shape[0]):
            if MSA[index_try,site_i] == 0:
                compteur+=1
                
        Frequence_1body[site_i,0] = pseudocount/2 + (1-pseudocount)*compteur/MSA.shape[0]
        Frequence_1body[site_i,1] = 1 - Frequence_1body[site_i,0]
        
    for site_i in range(middle_index):
        for site_j in range(middle_index,MSA.shape[1]):
            for index_spin_1 in range(2):
                for index_spin_2 in range(2):
                    PMI[site_i,site_j,index_spin_1,index_spin_2] = log(Fij[site_i,site_j,index_spin_1,index_spin_2]/(Frequence_1body[site_i,index_spin_1]*Frequence_1body[site_j,index_spin_2]))
            
    return np.asarray(PMI)