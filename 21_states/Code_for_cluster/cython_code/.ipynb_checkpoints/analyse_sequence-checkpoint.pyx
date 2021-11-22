# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False

import numpy as np
cimport cython
from cython.parallel import prange
        
def Cij_cython(char[:,::1] msa,double pseudocount, int n_states, double theta):
    """
    Return a Matrix of correleation with q-1 state (the last correlation because of gauge chosen is eij(q,)= 0)
    """
    cdef :
        int site_i,site_j,index_spin_1,index_spin_2,n_states_gauge = int(n_states-1)
        ##PMI[site_i,site_j,spin_i,spin_j]
        double[::1,:,:,:] Cij = np.zeros((msa.shape[1],n_states_gauge,msa.shape[1],n_states_gauge),order = "F")
        double[::1,:,:,:] Fij
        double[::1] weight 
    ##Cython
    Fij, weight = two_body_freq_cython(msa, pseudocount, n_states, theta)
    ##
    with nogil:
        for site_i in prange(msa.shape[1]):
            for site_j in range(msa.shape[1]):
                for index_spin_1 in range(n_states_gauge):
                    for index_spin_2 in range(n_states_gauge):
                        Cij[site_i,index_spin_1,site_j,index_spin_2] = Fij[site_i,index_spin_1,site_j,index_spin_2]-Fij[site_i,index_spin_1,site_i,index_spin_1]*Fij[site_j,index_spin_2,site_j,index_spin_2]   
    return np.asarray(Cij), weight

def two_body_freq_cython(char[:,::1] msa, double pseudocount, int n_states, double theta):
    cdef:
        double[::1,:,:,:] Fij = np.zeros((msa.shape[1],n_states,msa.shape[1],n_states),order = "F")
        int site_i,site_j,index_try,index_spin_1,index_spin_2
        double z
        double[::1] weight 
    weight = weight_msa(msa, theta)
    z = np.sum(weight)
    with nogil:
        for site_i in prange(msa.shape[1]):
            for site_j in range(msa.shape[1]):
                for index_try in range(msa.shape[0]):
                    Fij[site_i,msa[index_try,site_i],site_j, msa[index_try,site_j]] += weight[index_try]/z 
                for index_spin_1 in range(n_states):
                    for index_spin_2 in range(n_states):
                        if site_i != site_j:
                            Fij[site_i,index_spin_1,site_j,index_spin_2] = pseudocount/(n_states**2) + (1-pseudocount)*Fij[site_i,index_spin_1,site_j,index_spin_2]
                        else:
                            Fij[site_i,index_spin_1,site_j,index_spin_2] = (index_spin_1 == index_spin_2)*1*(pseudocount/n_states) + (1-pseudocount)*Fij[site_i,index_spin_1,site_j,index_spin_2]
    return np.asarray(Fij),weight

cpdef double[::1] weight_msa(char[:,::1] msa, double theta):
    cdef:
        double[::1] weight = np.ones((msa.shape[0]))
        double[::1] sum_dist = np.zeros((msa.shape[0]))
        int i,j
    if theta>0:
        with nogil:
            for i in prange(msa.shape[0]-1):
                for j in range(i+1,msa.shape[0]):
                    if hamming_dist(msa[i],msa[j])<theta:
                        sum_dist[i] += 1
                        sum_dist[j] += 1
    for i in range(msa.shape[0]):
        weight[i] = 1/(sum_dist[i]+1) #dont take into account the diagonal
    return weight

cdef double hamming_dist(char[::1] a, char[::1] b) nogil:
    cdef:
        int i
        double dist = 0.0
    for i in range(a.shape[0]):
        dist+= (a[i]!=b[i])*1.0
    return dist/a.shape[0]
        
def Energy_Partner(char[:,::1] MSA_Testing, double[:,:,:,::1] Jij, int middle_index):
    """
    Input :
    MSA_Testing
    Jij = Coupling inferred on MSA_training
    Output 
    Eij = Energy protein in MSA_L1 with protein in MSA_L2
    """
    cdef:
        int protein1, protein2, index_site_i,index_site_j
        double[:,::1] Eij = np.zeros((MSA_Testing.shape[0],MSA_Testing.shape[0]))
    with nogil:
        for protein1 in prange(MSA_Testing.shape[0]):
            for protein2 in range(MSA_Testing.shape[0]):
                for index_site_i in range(middle_index):
                    for index_site_j in range(middle_index,MSA_Testing.shape[1]):
                        Eij[protein1,protein2] -= Jij[index_site_i,MSA_Testing[protein1,index_site_i],index_site_j,MSA_Testing[protein2,index_site_j]]           
    return np.asarray(Eij)

def Ising_gauge_Jij(double[::1,:,:,:] Jij):
    """
    In : Jij L*(q-1) gauge Jij[:,q] = 0
    Out : Jij L*q gauge Ising
    """
    cdef:
        int index_i, index_j, state_1, state_2, n_states_gauge = Jij.shape[1]
        double mean_second_site, mean_first_site, mean_two_sites
        double[:,:,:,::1] Jij_out = np.zeros((Jij.shape[0],Jij.shape[1]+1,Jij.shape[2],Jij.shape[3]+1))
    with nogil:
        for index_i in prange(Jij.shape[0]):
            for index_j in range(Jij.shape[2]):
                mean_two_sites = mean_list_2(Jij,index_i,index_j)
                for state_1 in range(n_states_gauge):
                    mean_second_site = mean_list(Jij,index_i,state_1,index_j)
                    for state_2 in range(n_states_gauge):
                        mean_first_site = mean_list_1(Jij,index_i,index_j,state_2)
                        Jij_out[index_i,state_1,index_j,state_2] = Jij[index_i,state_1,index_j,state_2] - mean_first_site - mean_second_site + mean_two_sites
    return np.asarray(Jij_out)

cdef double mean_list(double[::1,:,:,:] List, int index_fix, int index_fix_2,int index_fix_3 ) nogil:
    cdef:
        double mean = 0.0
        int index_list
    for index_list in range(List.shape[3]):
        mean += List[index_fix,index_fix_2,index_fix_3,index_list]
    return mean/(List.shape[3]+1) #zero of gauge excluded in list

cdef double mean_list_1(double[::1,:,:,:] List, int index_fix, int index_fix_2, int index_fix_3 ) nogil:
    cdef:
        double mean = 0.0
        int index_list
    for index_list in range(List.shape[1]):
        mean += List[index_fix,index_list,index_fix_2,index_fix_3]
    return mean/(List.shape[1]+1)#zero of gauge excluded in list

cdef double mean_list_2(double[::1,:,:,:] List, int index_fix_0,int index_fix_2) nogil:
    cdef:
        double mean = 0.0
        int index_list, index_list_2
    for index_list in range(List.shape[1]):
        for index_list_2 in range(List.shape[3]):
            mean += List[index_fix_0,index_list,index_fix_2,index_list_2]
    return mean/((List.shape[1]+1)*(List.shape[3]+1))#zero of gauge excluded in list