from scipy.optimize import linear_sum_assignment
from scipy import linalg
import cython_code.analyse_sequence as an
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import fasta_file.extract_msa as extr


def Inference_Partner_real_data(msa, l_species, nb_paires, regularisation, n_state_spin, middle_index, theta):
    percent_true_partner = 0
    counter_average = 0
    m_size = 0
    l_m_size = []
    n_avg = 50
    size_train=0
    for avg in range(n_avg):
        l_per = np.random.permutation(len(l_species))
        l_in = np.random.permutation(msa.shape[0])
        msa_train = []
        i_species=0
        while len(msa_train)<nb_paires:
            for j in l_species[l_per[i_species]]:
                msa_train.append(msa[j])
                size_train += 1
                if len(msa_train)>=nb_paires:
                    break
            i_species+=1
        msa_train = np.array(msa_train,dtype=np.int8)  
        L = msa_train.shape[1]
        Cij_shape4,weight = an.Cij_cython(msa_train, regularisation, n_state_spin, theta)
        Cij = np.reshape(Cij_shape4,(L*(n_state_spin-1),L*(n_state_spin-1)),order = "F") # F contigunous array for speed up the inverse function wrote in fortran
        linalg.inv(Cij, overwrite_a=True)
        Jij_q_1 =  np.reshape(-1*Cij,(L,n_state_spin-1,L,n_state_spin-1),order = "F")
        Jij = an.Ising_gauge_Jij(Jij_q_1) #C contiguous array now !!!
        ### 
        ind_species = i_species#len(l_species)-100 ## Test on all species
        while ind_species<len(l_species):
            msa_test=np.array(msa[l_species[l_per[ind_species]]],dtype=np.int8)
            Cost = an.Energy_Partner(msa_test, Jij, middle_index)
            Permutation_worker_row = np.random.permutation(Cost.shape[0])
            Cost = Cost[Permutation_worker_row]
            row_ind, col_ind = linear_sum_assignment(Cost)
            percent_true_partner += np.mean(Permutation_worker_row[row_ind] == col_ind) # The true index of worker i is Permutation_worker[i]
            m_size += len(Permutation_worker_row)
            l_m_size.append(len(Permutation_worker_row))
            counter_average += 1
            ind_species+=1      
    return percent_true_partner/counter_average,m_size/counter_average,size_train/n_avg,l_m_size

    
file_fasta = "fasta_file/Concat_nnn_extr_withFirst.fasta"
d_species = extr.dictionnary_species(file_fasta)
msa = extr.get_msa_fasta_file(file_fasta)

l_species = [l for l in list(d_species.values()) if len(l)>=2]

nb_paires_max = 3000
regularisation = 0.5
theta = 0.3
middle_index = 63 #Split between first and second protein 
n_state_spin = 21
l_size_training = [];l_plot = [];l_size_testing_set = []
threshold_print = 0
l_nb_paires = np.unique(np.geomspace(1,nb_paires_max,num=20,dtype=int))
for ind,nb_paires in enumerate(l_nb_paires):
    #if nb_species/nb_species_train>threshold_print:
       # threshold_print += 0.2
    #print(ind/len(l_nb_paires)*100," %")
    tp,m_size,size_train,l_m_size = Inference_Partner_real_data(msa
                            , l_species
                            , nb_paires
                            , regularisation
                            , n_state_spin
                            , middle_index
                            , theta)
    l_size_testing_set.extend(l_m_size)
    l_size_training.append(size_train)
    l_plot.append(tp)


np.save("data/l_size_testing_set",l_size_testing_set)
np.save("data/l_size_training",l_size_training)
np.save("data/l_plot",l_plot)
np.save("data/l_nb_paires",l_nb_paires)
