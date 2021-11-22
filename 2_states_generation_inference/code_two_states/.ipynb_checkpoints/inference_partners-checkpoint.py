import numpy as np
from scipy.optimize import linear_sum_assignment
import generation_sequences as ge 

def Inference_Partner(l_msa, s_train, reg, n_pair, theta=0.2, fast=False, middle_index=None):
    M_MSA = l_msa
    Size_Training = s_train
    regularisation = reg
    number_of_pair_in_species = n_pair
    n_mean = 20
    Percentage_True_partner = np.zeros(number_of_pair_in_species)
    counter_average = 0
    for index_mean in range(n_mean):
        for index_test in range(M_MSA.shape[0]):
            Liste_permutation = np.random.permutation(M_MSA.shape[1])
            MSA_Training = M_MSA[index_test,Liste_permutation[:Size_Training]]
            if fast :
                index_max = min(Size_Training + 50*number_of_pair_in_species,np.size(M_MSA,axis=1))
                MSA_Testing = M_MSA[index_test,Liste_permutation[Size_Training:index_max]]
            else :
                MSA_Testing = M_MSA[index_test,Liste_permutation[Size_Training:]]
            Jij = Inference_Jij(MSA_Training,regularisation,theta)
            Eij = Energy_Partner(MSA_Testing,Jij,middle_index)
            ind_species = number_of_pair_in_species
            while ind_species<MSA_Testing.shape[0]:
                Cost = Eij[ind_species-number_of_pair_in_species:ind_species,ind_species-number_of_pair_in_species:ind_species]
                Permutation_worker_row = np.random.permutation(Cost.shape[0])
                Cost = Cost[Permutation_worker_row]
                row_ind, col_ind = linear_sum_assignment(Cost)
                Percentage_True_partner += Permutation_worker_row[row_ind] == col_ind # The true index of worker i is Permutation_worker[i]
                counter_average += 1
                ind_species+=number_of_pair_in_species
    return np.mean(Percentage_True_partner/counter_average)


def Energy_Partner(MSA_Testing, Jij, middle_index):
    """
    Input :
    MSA_Testing
    Jij = Coupling inferred on MSA_training
    Output 
    Eij = Energy protein in MSA_L1 with protein in MSA_L2
    """
    if middle_index is None:
        middle_index = int(MSA_Testing.shape[1]/2)
    MSA_L1,MSA_L2 = MSA_Testing[:,:middle_index], MSA_Testing[:,middle_index:]
    M_conca_r = np.zeros((MSA_Testing.shape[0], MSA_L1.shape[1] + MSA_L2.shape[1]))
    M_conca_l = np.zeros((MSA_Testing.shape[0], MSA_L1.shape[1] + MSA_L2.shape[1]))
    M_conca_r[:,:MSA_L1.shape[1]] = MSA_L1
    M_conca_l[:,MSA_L1.shape[1]:] = MSA_L2
    Eij = - np.matmul(M_conca_l,np.matmul(Jij,M_conca_r.T))
    return Eij     
           
def Inference_Jij(MSA_Training,regularisation,theta):
    Cij_regularized = regularisation_MSA(MSA_Training,regularisation,theta) 
    Jij = - np.linalg.inv(Cij_regularized)
    return Jij
                
def regularisation_MSA(MSA_input, regularisation, theta):
    """
    MSA : database of list of realisation of spin  [[one realisation],[an other], ] (Conversion from bool to -1 +1 if type is bool)
    regularisation : pourcentage of random chain of spin add to make Corelation matrix invertible (lambda)
    return : Correlation_Matrix regularized by if i!=j : C_ij = (1-lambda)<\sigma_i \sigma_j> - (1-lambda)**2<\sigma_i> <\sigma_j> = (1-lambda)*C_ij(sans regu) + ((1-lambda) - (1-lambda)**2) * Matrix(<\sigma_i><\sigma_j>)
    if i=j : C_ii = lambda + (1-lambda)<\sigma_i \sigma_i> - (1-lambda)**2<\sigma_i>**2 
    """
    if MSA_input.dtype == "bool":
        MSA = (MSA_input*1.0 - 0.5)*2.0
    else:
        MSA = MSA_input
    MSA = MSA.T
    weights = ge.weight_msa(MSA.T,theta)
    avg_a = np.average(MSA,axis = 1, weights=weights)
    Vector_matrix = np.tile(avg_a,(len(avg_a),1))
    Id_avg = np.diagflat(avg_a)
    Matrix_product_avg = Id_avg@Vector_matrix
    Cij = np.cov(MSA, aweights=weights)
    C_regularized = (1-regularisation)*(Cij+ regularisation* Matrix_product_avg) + regularisation*np.identity(len(avg_a))
    return C_regularized

def triangularise(m):
    """
    Put -inf in the diagonale and below it
    """
    m = np.asanyarray(m)
    mask = np.tri(*m.shape[-2:], k=0, dtype=bool)
    return np.where(mask, -1*np.ones(1, m.dtype)*np.inf, m)