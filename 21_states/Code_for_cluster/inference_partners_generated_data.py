import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import linalg
import cython_code.analyse_sequence as an
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import fasta_file.extract_msa as extr
plt.rcParams["figure.figsize"] = (10,5)

def inference_partner_real_data(msa, l_species, nb_paires, regularisation, n_state_spin, middle_index, theta):
    percent_true_partner = 0
    counter_average = 0
    m_size = 0
    n_avg = 20
    for avg in range(n_avg):
        l_per = np.random.permutation(len(l_species))
        l_in = np.random.permutation(msa.shape[0])
        msa_train = []
        i_species=0
        while len(msa_train)<nb_paires:
            for j in l_species[l_per[i_species]]:
                msa_train.append(msa[j])
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
            msa_test = np.array(msa[l_species[l_per[ind_species]]],dtype=np.int8)
            Cost = an.Energy_Partner(msa_test, Jij, middle_index)
            Permutation_worker_row = np.random.permutation(Cost.shape[0])
            Cost = Cost[Permutation_worker_row]
            row_ind, col_ind = linear_sum_assignment(Cost)
            percent_true_partner += np.mean(Permutation_worker_row[row_ind] == col_ind) # The true index of worker i is Permutation_worker[i]
            m_size += len(Permutation_worker_row)
            counter_average += 1
            ind_species += 1
    return percent_true_partner/counter_average,m_size/counter_average

def l_plot_inference_real_data(msa, l_species, l_size_train, regularisation, n_state_spin, middle_index, theta):
    l_plot = []
    for ind,nb_paires in enumerate(l_size_train):
        print(ind/len(l_size_train)*100," %")
        tp,_ = inference_partner_real_data(msa, l_species, nb_paires, regularisation, n_state_spin, middle_index, theta)
        l_plot.append(tp)
    return l_plot

def inference_partner_generated(msa, size_train, number_of_pair_in_species, 
                        regularisation, n_state_spin, middle_index, theta):
    percent_true_partner = 0
    counter_average = 0
    n_avg = 20
    for avg in range(n_avg):
        Liste_permutation = np.random.permutation(msa.shape[0])
        msa_train = msa[Liste_permutation[:size_train]]
        msa_testing = msa[Liste_permutation[size_train:]]
        L = msa_train.shape[1]
        Cij_shape4,weight = an.Cij_cython(msa_train, regularisation, n_state_spin, theta)
        Cij = np.reshape(Cij_shape4,(L*(n_state_spin-1),L*(n_state_spin-1)),order = "F") # F contigunous array for speed up the inverse function wrote in fortran
        linalg.inv(Cij, overwrite_a=True)
        Jij_q_1 =  np.reshape(-1*Cij,(L,n_state_spin-1,L,n_state_spin-1),order = "F")
        Jij = an.Ising_gauge_Jij(Jij_q_1) #C contiguous array now !!!
        ### 
        ind_species = number_of_pair_in_species
        while ind_species<msa_testing.shape[0]:
            msa_test = msa_testing[ind_species - number_of_pair_in_species:ind_species]
            Cost = an.Energy_Partner(msa_test, Jij, middle_index)
            Permutation_worker_row = np.random.permutation(Cost.shape[0])
            Cost = Cost[Permutation_worker_row]
            row_ind, col_ind = linear_sum_assignment(Cost)
            percent_true_partner += np.sum(Permutation_worker_row[row_ind] == col_ind) # The true index of worker i is Permutation_worker[i]
            counter_average += 1*msa_test.shape[0]
            ind_species += number_of_pair_in_species  
    return percent_true_partner/counter_average

def l_plot_inference(msa,l_size_train,number_of_pair_in_species,regularisation,n_state_spin,middle_index,theta):
    l_plot = []
    for ind,size_train in enumerate(l_size_train):
        print(ind/len(l_size_train)*100," %")
        tp = inference_partner_generated(msa, size_train, number_of_pair_in_species, 
            regularisation, n_state_spin, middle_index, theta
            )
        l_plot.append(tp)
    return l_plot

size_train_max = 5000
number_of_pair_in_species = 11 
regularisation = 0.5
theta = 0.0
n_state_spin = 21
middle_index = 63
file_fasta = "fasta_file/Concat_nnn_withFirst.fasta"

msa = extr.get_msa_fasta_file(file_fasta)
d_species = extr.dictionnary_species(file_fasta)
l_species = [l for l in list(d_species.values()) if len(l)>=2]

msa_no_phylo_bmdca = np.load("data_bmdca/msa_no_phylo_bmdca.npy")
msa_phylo_tree_bmdca = np.load("data_bmdca/msa_phylo_bmdca_tree.npy")
msa_phylo_equi_tree_bmdca = np.load("data_bmdca/msa_phylo_equi_bmdca_tree.npy")

msa_no_phylo_ardca = np.load("data_ardca/msa_no_phylo_ardca.npy")
msa_phylo_tree_ardca = np.load("data_ardca/msa_phylo_ardca_tree.npy")
msa_phylo_equi_tree_ardca = np.load("data_ardca/msa_phylo_equi_ardca_tree.npy") 

l_size_train = np.unique(np.geomspace(1,size_train_max,num=30,dtype=int))
args = [l_size_train,number_of_pair_in_species,regularisation,n_state_spin,middle_index,theta]
args_real_data = [l_species, l_size_train, regularisation, n_state_spin, middle_index, theta]

l_plot = l_plot_inference(msa, *args)
l_plot_real_species = l_plot_inference_real_data(msa, *args_real_data)

l_plot_no_phylo_bmdca = l_plot_inference_real_data(msa_no_phylo_bmdca, *args_real_data)
l_plot_phylo_tree_bmdca = l_plot_inference_real_data(msa_phylo_tree_bmdca, *args_real_data)
l_plot_phylo_equi_tree_bmdca = l_plot_inference_real_data(msa_phylo_equi_tree_bmdca, *args_real_data)

l_plot_no_phylo_ardca = l_plot_inference_real_data(msa_no_phylo_ardca, *args_real_data)
l_plot_phylo_tree_ardca = l_plot_inference_real_data(msa_phylo_tree_ardca, *args_real_data)
l_plot_phylo_equi_tree_ardca = l_plot_inference_real_data(msa_phylo_equi_tree_ardca, *args_real_data)

np.save("output_inference_partners_generated/l_size_train",l_size_train)
np.save("output_inference_partners_generated/l_plot",l_plot)
np.save("output_inference_partners_generated/l_plot_real_species",l_plot_real_species)

np.save("output_inference_partners_generated/l_plot_no_phylo_bmdca",l_plot_no_phylo_bmdca)
np.save("output_inference_partners_generated/l_plot_phylo_tree_bmdca",l_plot_phylo_tree_bmdca)
np.save("output_inference_partners_generated/l_plot_phylo_equi_tree_bmdca",l_plot_phylo_equi_tree_bmdca)

np.save("output_inference_partners_generated/l_plot_no_phylo_ardca",l_plot_no_phylo_ardca)
np.save("output_inference_partners_generated/l_plot_phylo_tree_ardca",l_plot_phylo_tree_ardca)
np.save("output_inference_partners_generated/l_plot_phylo_equi_tree_ardca",l_plot_phylo_equi_tree_ardca)
