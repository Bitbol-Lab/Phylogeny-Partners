import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import linalg
import cython_code.analyse_sequence as an
import numpy as np
import matplotlib.pyplot as plt
import utils_fasta_file as extr

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
            msa_test=np.array(msa[l_species[l_per[ind_species]]],dtype=np.int8)
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
        if ind%int(len(l_size_train)/5) == 0:
            print(ind/len(l_size_train)*100," %")
        tp,_ = inference_partner_real_data(msa, l_species, nb_paires, regularisation, n_state_spin, middle_index, theta)
        l_plot.append(tp)
    return l_plot

def inference_partner_generated(msa, size_train, number_of_pair_in_species, 
                        regularisation, n_state_spin, middle_index, theta, graph=None, Jij_para=None, msa_testing_input=None):
    percent_true_partner = 0
    counter_average = 0
    n_avg = 30 if msa_testing_input is None else 1
    
    for avg in range(n_avg):
        Liste_permutation = np.random.permutation(msa.shape[0])
        msa_train = msa[Liste_permutation[:size_train]]
        msa_testing = msa[Liste_permutation[size_train:]] 

        if Jij_para is None:
            Jij = estimation_J_ij(msa_train, regularisation, n_state_spin, theta)
            if not graph is None:
                Jij = true_contact_Jij(Jij, graph)
        else:
            Jij = Jij_para
            if not msa_testing_input is None:
                msa_testing = msa_testing_input
            if not graph is None:
                Jij = true_contact_Jij(Jij, graph)
                
        ind_species = number_of_pair_in_species
        while ind_species<msa_testing.shape[0]:
            msa_test = msa_testing[ind_species - number_of_pair_in_species:ind_species]
            Cost = an.Energy_Partner(msa_test, Jij, middle_index)
            Permutation_worker_row = np.random.permutation(Cost.shape[0])
            Cost = Cost[Permutation_worker_row]
            row_ind, col_ind = linear_sum_assignment(Cost)
            percent_true_partner += np.mean(Permutation_worker_row[row_ind] == col_ind) # The true index of worker i is Permutation_worker[i]
            counter_average += 1
            ind_species += number_of_pair_in_species
    return percent_true_partner/counter_average, Jij

def l_plot_inference(msa, l_size_train, number_of_pair_in_species, regularisation, n_state_spin, middle_index, theta):
    l_plot = []
    l_Jij = []
    for ind,size_train in enumerate(l_size_train):
        if ind%int(len(l_size_train)/5) == 0:
            print(ind/len(l_size_train)*100," %")
        tp, Jij_mean = inference_partner_generated(msa, size_train, number_of_pair_in_species, 
            regularisation, n_state_spin, middle_index, theta
            )
        l_plot.append(tp)
        l_Jij.append(Jij_mean)
    return l_plot, l_Jij

def l_plot_inference_true_contact(msa, l_size_train, number_of_pair_in_species, regularisation, n_state_spin, middle_index, theta, graph):
    l_plot = []
    l_Jij = []
    for ind,size_train in enumerate(l_size_train):
        if ind%int(len(l_size_train)/5) == 0:
            print(ind/len(l_size_train)*100," %")
        tp, Jij_mean = inference_partner_generated(msa, size_train, number_of_pair_in_species, 
            regularisation, n_state_spin, middle_index, theta, graph=graph
            )
        l_plot.append(tp)
        l_Jij.append(Jij_mean)
    return l_plot, l_Jij

def estimation_J_ij(msa, regularisation, n_state_spin, theta):
    L = msa.shape[1]
    Cij_shape4, weight = an.Cij_cython(msa, regularisation, n_state_spin, theta)
    Cij = np.reshape(Cij_shape4,(L*(n_state_spin-1),L*(n_state_spin-1)),order = "F") # F contigunous array for speed up the inverse function wrote in fortran
    linalg.inv(Cij, overwrite_a=True)
    Jij_q_1 =  np.reshape(-1*Cij,(L,n_state_spin-1,L,n_state_spin-1),order = "F")
    Jij = an.Ising_gauge_Jij(Jij_q_1) #C contiguous array now !!!
    return Jij

def true_contact_Jij(Jij, graph):
    t_edge = graph.edges()
    Jij_r = Jij#np.copy(Jij)
    for i in range(Jij.shape[0]-1):
        for j in range(i+1,Jij.shape[2]):
            if not (i,j) in t_edge:
                Jij_r[i,:,j,:]=np.zeros((Jij.shape[1],Jij.shape[3]))
                Jij_r[j,:,i,:]=np.zeros((Jij.shape[1],Jij.shape[3]))
    return Jij_r


"""
            Cij_shape4, weight = an.Cij_cython(msa_train, regularisation, n_state_spin, theta)
            Cij = np.reshape(Cij_shape4,(L*(n_state_spin-1),L*(n_state_spin-1)),order = "F") # F contigunous array for speed up the inverse function wrote in fortran
            linalg.inv(Cij, overwrite_a=True)
            Jij_q_1 =  np.reshape(-1*Cij,(L,n_state_spin-1,L,n_state_spin-1),order = "F")
            Jij = an.Ising_gauge_Jij(Jij_q_1) #C contiguous array now !!!
"""