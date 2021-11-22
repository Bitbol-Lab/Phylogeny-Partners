import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import linalg
import cython_code.analyse_sequence as an
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fasta_file.extract_msa as extr
import networkx as nx
plt.rcParams["figure.figsize"] = (10,5)

def inference_partner_real_data(msa, l_species, nb_paires, regularisation, n_state_spin, middle_index, theta, graph, false_contact):
    percent_true_partner = 0
    counter_average = 0
    m_size = 0
    n_avg = 10
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
        Cij_shape4, weight = an.Cij_cython(msa_train, regularisation, n_state_spin, theta)
        Cij = np.reshape(Cij_shape4,(L*(n_state_spin-1),L*(n_state_spin-1)),order = "F") # F contigunous array for speed up the inverse function wrote in fortran
        linalg.inv(Cij, overwrite_a=True)
        Jij_q_1 =  np.reshape(-1*Cij,(L,n_state_spin-1,L,n_state_spin-1),order = "F")
        Jij = an.Ising_gauge_Jij(Jij_q_1) #C contiguous array now !!!
        ### MODIF JIJ to exclude some false coupling #####
        if false_contact:
            Jij = false_contact_Jij(Jij, graph)
        else:
            Jij = true_contact_Jij(Jij, graph)
        ################################################
        ind_species = i_species #len(l_species)-100 ## Test on all species
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

def l_plot_inference_real_data(msa, l_species, l_size_train, regularisation, n_state_spin, middle_index, theta, graph, false_contact=False):
    l_plot = []
    for ind,nb_paires in enumerate(l_size_train):
        print(ind/len(l_size_train)*100," %")
        tp,_ = inference_partner_real_data(msa, l_species, nb_paires, regularisation, n_state_spin, middle_index, theta, graph, false_contact)
        l_plot.append(tp)
    return l_plot

def true_contact_Jij(Jij, graph):
    t_edge = graph.edges()
    for i in range(Jij.shape[0]-1):
        for j in range(i+1,Jij.shape[2]):
            if not (i,j) in t_edge:
                Jij[i,:,j,:]=np.zeros((Jij.shape[1],Jij.shape[3]))
                Jij[j,:,i,:]=np.zeros((Jij.shape[1],Jij.shape[3]))
    return Jij

def false_contact_Jij(Jij, graph):
    t_edge = graph.edges()
    for i in range(Jij.shape[0]-1):
        for j in range(i+1,Jij.shape[2]):
            if (i,j) in t_edge:
                Jij[i,:,j,:]=np.zeros((Jij.shape[1],Jij.shape[3]))
                Jij[j,:,i,:]=np.zeros((Jij.shape[1],Jij.shape[3]))
    return Jij

size_train_max = 5000
regularisation = 0.5
theta = 0.0
n_state_spin = 21
middle_index = 63
file_fasta = "fasta_file/Concat_nnn_withFirst.fasta"
Graph_HK_RR_4 = nx.read_gexf("Extract_contact_pdb_HK-RR/Graph_HK_and_RR_Threshold_4", node_type = int)
Graph_HK_RR_8 = nx.read_gexf("Extract_contact_pdb_HK-RR/Graph_HK_and_RR_Threshold_8", node_type = int)

msa = extr.get_msa_fasta_file(file_fasta)
d_species = extr.dictionnary_species(file_fasta)
l_species = [l for l in list(d_species.values()) if len(l)>=2]

# msa_no_phylo_ccmpred = extr.get_msa_fasta_file("data_ccmpred/msa.mcmc_no_phylo.fas")
# msa_phylo_equi_tree_ccmpred = extr.get_msa_fasta_file("data_ccmpred/msa.mcmc_phylo_fast_tree_equi_mutation_rate_1.fas")
# msa_phylo_tree_ccmpred = extr.get_msa_fasta_file("data_ccmpred/msa.mcmc_phylo_fast_tree_random_mutation_rate_1.fas")

msa_no_phylo_bmdca = np.load("data_bmdca/msa_no_phylo_bmdca.npy")
msa_phylo_tree_bmdca = np.load("data_bmdca/msa_phylo_bmdca_tree.npy")
msa_phylo_equi_tree_bmdca = np.load("data_bmdca/msa_phylo_equi_bmdca_tree.npy")

msa_no_phylo_ardca = np.load("data_ardca/msa_no_phylo_ardca.npy")
msa_phylo_tree_ardca = np.load("data_ardca/msa_phylo_ardca_tree.npy")
msa_phylo_equi_tree_ardca = np.load("data_ardca/msa_phylo_equi_ardca_tree.npy") 

l_size_train = np.unique(np.geomspace(1,size_train_max,num=30,dtype=int))
args_real_data = [l_species, l_size_train, regularisation, n_state_spin, middle_index, theta, Graph_HK_RR_4]

l_plot_real_species = l_plot_inference_real_data(msa, *args_real_data)

# l_plot_no_phylo_ccmpred = l_plot_inference_real_data(msa_no_phylo_ccmpred, *args_real_data)
# l_plot_phylo_tree_ccmpred = l_plot_inference_real_data(msa_phylo_tree_ccmpred,*args_real_data)
# l_plot_phylo_equi_tree_ccmpred = l_plot_inference_real_data(msa_phylo_equi_tree_ccmpred,*args_real_data)

l_plot_no_phylo_bmdca = l_plot_inference_real_data(msa_no_phylo_bmdca, *args_real_data)
l_plot_phylo_tree_bmdca = l_plot_inference_real_data(msa_phylo_tree_bmdca, *args_real_data)
l_plot_phylo_equi_tree_bmdca = l_plot_inference_real_data(msa_phylo_equi_tree_bmdca, *args_real_data)

l_plot_no_phylo_ardca = l_plot_inference_real_data(msa_no_phylo_ardca, *args_real_data)
l_plot_phylo_tree_ardca = l_plot_inference_real_data(msa_phylo_tree_ardca, *args_real_data)
l_plot_phylo_equi_tree_ardca = l_plot_inference_real_data(msa_phylo_equi_tree_ardca, *args_real_data)

np.save("output_inference_partners_generated/l_size_train_graph_4",l_size_train)
np.save("output_inference_partners_generated/l_plot_real_species_graph_4",l_plot_real_species)

# np.save("output_inference_partners_generated/l_plot_no_phylo_ccmpred_graph_4",l_plot_no_phylo_ccmpred)
# np.save("output_inference_partners_generated/l_plot_phylo_tree_ccmpred_graph_4",l_plot_phylo_tree_ccmpred)
# np.save("output_inference_partners_generated/l_plot_phylo_equi_tree_ccmpred_graph_4",l_plot_phylo_equi_tree_ccmpred)

np.save("output_inference_partners_generated/l_plot_no_phylo_bmdca_graph_4",l_plot_no_phylo_bmdca)
np.save("output_inference_partners_generated/l_plot_phylo_tree_bmdca_graph_4",l_plot_phylo_tree_bmdca)
np.save("output_inference_partners_generated/l_plot_phylo_equi_tree_bmdca_graph_4",l_plot_phylo_equi_tree_bmdca)

np.save("output_inference_partners_generated/l_plot_no_phylo_ardca_graph_4",l_plot_no_phylo_ardca)
np.save("output_inference_partners_generated/l_plot_phylo_tree_ardca_graph_4",l_plot_phylo_tree_ardca)
np.save("output_inference_partners_generated/l_plot_phylo_equi_tree_ardca_graph_4",l_plot_phylo_equi_tree_ardca)


args_real_data = [l_species, l_size_train, regularisation, n_state_spin, middle_index, theta, Graph_HK_RR_8]

l_plot_real_species = l_plot_inference_real_data(msa, *args_real_data)

# l_plot_no_phylo_ccmpred = l_plot_inference_real_data(msa_no_phylo_ccmpred, *args_real_data)
# l_plot_phylo_tree_ccmpred = l_plot_inference_real_data(msa_phylo_tree_ccmpred,*args_real_data)
# l_plot_phylo_equi_tree_ccmpred = l_plot_inference_real_data(msa_phylo_equi_tree_ccmpred,*args_real_data)

l_plot_no_phylo_bmdca = l_plot_inference_real_data(msa_no_phylo_bmdca, *args_real_data)
l_plot_phylo_tree_bmdca = l_plot_inference_real_data(msa_phylo_tree_bmdca, *args_real_data)
l_plot_phylo_equi_tree_bmdca = l_plot_inference_real_data(msa_phylo_equi_tree_bmdca, *args_real_data)

l_plot_no_phylo_ardca = l_plot_inference_real_data(msa_no_phylo_ardca, *args_real_data)
l_plot_phylo_tree_ardca = l_plot_inference_real_data(msa_phylo_tree_ardca, *args_real_data)
l_plot_phylo_equi_tree_ardca = l_plot_inference_real_data(msa_phylo_equi_tree_ardca, *args_real_data)


np.save("output_inference_partners_generated/l_size_train_graph_8",l_size_train)
np.save("output_inference_partners_generated/l_plot_real_species_graph_8",l_plot_real_species)

# np.save("output_inference_partners_generated/l_plot_no_phylo_ccmpred_graph_8",l_plot_no_phylo_ccmpred)
# np.save("output_inference_partners_generated/l_plot_phylo_tree_ccmpred_graph_8",l_plot_phylo_tree_ccmpred)
# np.save("output_inference_partners_generated/l_plot_phylo_equi_tree_ccmpred_graph_8",l_plot_phylo_equi_tree_ccmpred)

np.save("output_inference_partners_generated/l_plot_no_phylo_bmdca_graph_8",l_plot_no_phylo_bmdca)
np.save("output_inference_partners_generated/l_plot_phylo_tree_bmdca_graph_8",l_plot_phylo_tree_bmdca)
np.save("output_inference_partners_generated/l_plot_phylo_equi_tree_bmdca_graph_8",l_plot_phylo_equi_tree_bmdca)

np.save("output_inference_partners_generated/l_plot_no_phylo_ardca_graph_8",l_plot_no_phylo_ardca)
np.save("output_inference_partners_generated/l_plot_phylo_tree_ardca_graph_8",l_plot_phylo_tree_ardca)
np.save("output_inference_partners_generated/l_plot_phylo_equi_tree_ardca_graph_8",l_plot_phylo_equi_tree_ardca)


##############################33 False Contact ###############################################

args_real_data = [l_species, l_size_train, regularisation, n_state_spin, middle_index, theta, Graph_HK_RR_4]

l_plot_real_species = l_plot_inference_real_data(msa, *args_real_data, false_contact=True)

# l_plot_no_phylo_ccmpred = l_plot_inference_real_data(msa_no_phylo_ccmpred, *args_real_data, false_contact=True)
# l_plot_phylo_tree_ccmpred = l_plot_inference_real_data(msa_phylo_tree_ccmpred,*args_real_data, false_contact=True)
# l_plot_phylo_equi_tree_ccmpred = l_plot_inference_real_data(msa_phylo_equi_tree_ccmpred,*args_real_data, false_contact=True)

l_plot_no_phylo_bmdca = l_plot_inference_real_data(msa_no_phylo_bmdca, *args_real_data, false_contact=True)
l_plot_phylo_tree_bmdca = l_plot_inference_real_data(msa_phylo_tree_bmdca, *args_real_data, false_contact=True)
l_plot_phylo_equi_tree_bmdca = l_plot_inference_real_data(msa_phylo_equi_tree_bmdca, *args_real_data, false_contact=True)

l_plot_no_phylo_ardca = l_plot_inference_real_data(msa_no_phylo_ardca, *args_real_data, false_contact=True)
l_plot_phylo_tree_ardca = l_plot_inference_real_data(msa_phylo_tree_ardca, *args_real_data, false_contact=True)
l_plot_phylo_equi_tree_ardca = l_plot_inference_real_data(msa_phylo_equi_tree_ardca, *args_real_data, false_contact=True)


np.save("output_inference_partners_generated/l_plot_real_species_graph_4_false_contact=True",l_plot_real_species)

# np.save("output_inference_partners_generated/l_plot_no_phylo_ccmpred_graph_4_false_contact=True",l_plot_no_phylo_ccmpred)
# np.save("output_inference_partners_generated/l_plot_phylo_tree_ccmpred_graph_4_false_contact=True",l_plot_phylo_tree_ccmpred)
# np.save("output_inference_partners_generated/l_plot_phylo_equi_tree_ccmpred_graph_4_false_contact=True",l_plot_phylo_equi_tree_ccmpred)

np.save("output_inference_partners_generated/l_plot_no_phylo_bmdca_graph_4_false_contact=True",l_plot_no_phylo_bmdca)
np.save("output_inference_partners_generated/l_plot_phylo_tree_bmdca_graph_4_false_contact=True",l_plot_phylo_tree_bmdca)
np.save("output_inference_partners_generated/l_plot_phylo_equi_tree_bmdca_graph_4_false_contact=True",l_plot_phylo_equi_tree_bmdca)

np.save("output_inference_partners_generated/l_plot_no_phylo_ardca_graph_4_false_contact=True",l_plot_no_phylo_ardca)
np.save("output_inference_partners_generated/l_plot_phylo_tree_ardca_graph_4_false_contact=True",l_plot_phylo_tree_ardca)
np.save("output_inference_partners_generated/l_plot_phylo_equi_tree_ardca_graph_4_false_contact=True",l_plot_phylo_equi_tree_ardca)


args_real_data = [l_species, l_size_train, regularisation, n_state_spin, middle_index, theta, Graph_HK_RR_8]

l_plot_real_species = l_plot_inference_real_data(msa, *args_real_data, false_contact=True)

# l_plot_no_phylo_ccmpred = l_plot_inference_real_data(msa_no_phylo_ccmpred, *args_real_data, false_contact=True)
# l_plot_phylo_tree_ccmpred = l_plot_inference_real_data(msa_phylo_tree_ccmpred,*args_real_data, false_contact=True)
# l_plot_phylo_equi_tree_ccmpred = l_plot_inference_real_data(msa_phylo_equi_tree_ccmpred,*args_real_data, false_contact=True)

l_plot_no_phylo_bmdca = l_plot_inference_real_data(msa_no_phylo_bmdca, *args_real_data, false_contact=True)
l_plot_phylo_tree_bmdca = l_plot_inference_real_data(msa_phylo_tree_bmdca, *args_real_data, false_contact=True)
l_plot_phylo_equi_tree_bmdca = l_plot_inference_real_data(msa_phylo_equi_tree_bmdca, *args_real_data, false_contact=True)

l_plot_no_phylo_ardca = l_plot_inference_real_data(msa_no_phylo_ardca, *args_real_data, false_contact=True)
l_plot_phylo_tree_ardca = l_plot_inference_real_data(msa_phylo_tree_ardca, *args_real_data, false_contact=True)
l_plot_phylo_equi_tree_ardca = l_plot_inference_real_data(msa_phylo_equi_tree_ardca, *args_real_data, false_contact=True)


np.save("output_inference_partners_generated/l_plot_real_species_graph_8_false_contact=True",l_plot_real_species)

# np.save("output_inference_partners_generated/l_plot_no_phylo_ccmpred_graph_8_false_contact=True",l_plot_no_phylo_ccmpred)
# np.save("output_inference_partners_generated/l_plot_phylo_tree_ccmpred_graph_8_false_contact=True",l_plot_phylo_tree_ccmpred)
# np.save("output_inference_partners_generated/l_plot_phylo_equi_tree_ccmpred_graph_8_false_contact=True",l_plot_phylo_equi_tree_ccmpred)

np.save("output_inference_partners_generated/l_plot_no_phylo_bmdca_graph_8_false_contact=True",l_plot_no_phylo_bmdca)
np.save("output_inference_partners_generated/l_plot_phylo_tree_bmdca_graph_8_false_contact=True",l_plot_phylo_tree_bmdca)
np.save("output_inference_partners_generated/l_plot_phylo_equi_tree_bmdca_graph_8_false_contact=True",l_plot_phylo_equi_tree_bmdca)

np.save("output_inference_partners_generated/l_plot_no_phylo_ardca_graph_8_false_contact=True",l_plot_no_phylo_ardca)
np.save("output_inference_partners_generated/l_plot_phylo_tree_ardca_graph_8_false_contact=True",l_plot_phylo_tree_ardca)
np.save("output_inference_partners_generated/l_plot_phylo_equi_tree_ardca_graph_8_false_contact=True",l_plot_phylo_equi_tree_ardca)