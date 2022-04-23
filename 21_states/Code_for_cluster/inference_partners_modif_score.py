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
import argparse

def inference_partner_real_data(msa, l_species, nb_paires, regularisation, n_state_spin, middle_index, theta, graph, false_contact=None):
    print(f"theta = {theta}")
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
        msa_train = np.array(msa_train, dtype=np.int8)  
        L = msa_train.shape[1]
        Cij_shape4, weight = an.Cij_cython(msa_train, regularisation, n_state_spin, theta)
        #print(np.asarray(weight))
        Cij = np.reshape(Cij_shape4,(L*(n_state_spin-1),L*(n_state_spin-1)),order = "F") # F contigunous array for speed up the inverse function wrote in fortran
        linalg.inv(Cij, overwrite_a=True)
        Jij_q_1 =  np.reshape(-1*Cij,(L,n_state_spin-1,L,n_state_spin-1),order = "F")
        Jij = an.Ising_gauge_Jij(Jij_q_1) #C contiguous array now !!!
        ### MODIF JIJ to exclude some false coupling #####
        if false_contact is not None:
            if false_contact:
                Jij = false_contact_Jij(Jij, graph)
            else:
                Jij = true_contact_Jij(Jij, graph)
        ################################################
        ind_species = i_species #max(len(l_species)-10, i_species) #i_species #len(l_species)-100 ## Test on all species ##MODIF TO REMOVE
        while ind_species<len(l_species):
            msa_test = np.array(msa[l_species[l_per[ind_species]]],dtype=np.int8)
            Cost = an.Energy_Partner(msa_test, Jij, middle_index)
            Permutation_worker_row = np.random.permutation(Cost.shape[0])
            Cost = Cost[Permutation_worker_row]
            row_ind, col_ind = linear_sum_assignment(Cost)
            percent_true_partner += np.sum(Permutation_worker_row[row_ind] == col_ind) # The true index of worker i is Permutation_worker[i]
            m_size += len(Permutation_worker_row)
            counter_average += 1*msa_test.shape[0]
            ind_species += 1
    return percent_true_partner/counter_average,m_size/counter_average

def l_plot_inference_real_data(msa, l_species, l_size_train, regularisation, n_state_spin, middle_index, theta, graph, false_contact=None):
    l_plot = []
    for ind,nb_paires in enumerate(l_size_train):
        print(ind/len(l_size_train)*100," %")
        tp,_ = inference_partner_real_data(msa, l_species, nb_paires, regularisation, n_state_spin, middle_index, theta, graph, false_contact=false_contact)
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

def find_middle_index(Graph) -> int:
    for i in Graph.nodes(data="subset"):
        if i[1] == False:
            return i[0] - 1

def infer_partner(fasta_name, Graph_name, arDCA = True, theta = 0.0):
    regularisation = 0.5
    n_state_spin = 21
    Graph = nx.read_gexf("Extract_contact_pdb_HK-RR/%s"%Graph_name, node_type = int)
    middle_index = find_middle_index(Graph) # 63
    print(f"middle index : {middle_index}")
    
    file_fasta = "fasta_file/%s"%fasta_name
    msa = extr.get_msa_fasta_file(file_fasta)
    d_species = extr.dictionnary_species(file_fasta)
    
    l_species = [l for l in list(d_species.values()) if len(l)>=2]
    ltest = []
    for spe in l_species:
        ltest.extend(spe)

    print(f"Max i Species : {np.max(ltest)}")
    size_train_max = int(len(msa)/10)#5000
    print(f"size trainning set max : {size_train_max}, size msa : {len(msa)}")

    if not arDCA:
        print("Load bmDCA")
        name_program = "bmdca"
    else:
        print("Load arDCA")
        name_program = "ardca"

    msa_no_phylo = np.load(f"data_{name_program}/msa_no_phylo_{name_program}_{fasta_name}.npy")
    #msa_phylo_tree = np.load(f"data_{name_program}/msa_phylo_{fasta_name}_tree.npy")
    msa_phylo_equi_tree = np.load(f"data_{name_program}/msa_phylo_equi_{name_program}_tree_{fasta_name}.npy")

    l_size_train = np.unique(np.geomspace(1, size_train_max, num=30, dtype=int))
    args_real_data = [l_species, l_size_train, regularisation, n_state_spin, middle_index, theta, Graph]

    l_false_contact = [None, False, True]
    for false_contact in l_false_contact:
        np.save(f"output_inference_partners_generated/l_size_train_{name_program}_{Graph_name}_false_contact={false_contact}_theta={theta}", l_size_train)

        l_plot_real_species = l_plot_inference_real_data(msa, *args_real_data, false_contact=false_contact)
        np.save(f"output_inference_partners_generated/l_plot_real_species_{name_program}_{Graph_name}_false_contact={false_contact}_theta={theta}", l_plot_real_species)

        l_plot_no_phylo = l_plot_inference_real_data(msa_no_phylo, *args_real_data, false_contact=false_contact)
        np.save(f"output_inference_partners_generated/l_plot_no_phylo_{name_program}_{Graph_name}_false_contact={false_contact}_theta={theta}", l_plot_no_phylo)

        # l_plot_phylo_tree = l_plot_inference_real_data(msa_phylo_tree, *args_real_data, false_contact=false_contact)
        # np.save(f"output_inference_partners_generated/l_plot_phylo_tree_{name_program}_{Graph_name}_false_contact={false_contact}_theta={theta}", l_plot_phylo_tree)

        l_plot_phylo_equi_tree = l_plot_inference_real_data(msa_phylo_equi_tree, *args_real_data, false_contact=false_contact)
        np.save(f"output_inference_partners_generated/l_plot_phylo_equi_tree_{name_program}_{Graph_name}_false_contact={false_contact}_theta={theta}", l_plot_phylo_equi_tree)


# fasta_name = "MALG_MALK_cov75_hmmsearch_sorted_withLast_b.fas"
# Graph_name = "prot_MalG_MalK_Threshold_4_MinAllDist"
# infer_partner(fasta_name, Graph_name)

# Graph_name = "prot_MalG_MalK_Threshold_8_MinAllDist"
# infer_partner(fasta_name, Graph_name)

# Graph_name = "prot_MalG_MalK_Threshold_8_CarbonAlpha"
# infer_partner(fasta_name, Graph_name)

# fasta_name = "Concat_nnn_withFirst.fasta"
# Graph_name = "prot_HK_and_RR_Threshold_4_MinAllDist"
# infer_partner(fasta_name, Graph_name)
# infer_partner(fasta_name, Graph_name, arDCA=False)

# Graph_name = "prot_HK_and_RR_Threshold_8_MinAllDist"
# infer_partner(fasta_name, Graph_name)
# infer_partner(fasta_name, Graph_name, arDCA=False)

# Graph_name = "prot_HK_and_RR_Threshold_8_CarbonAlpha"
# infer_partner(fasta_name, Graph_name)
# infer_partner(fasta_name, Graph_name, arDCA=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta_name')
    parser.add_argument('graph_name')
    parser.add_argument('--arDCA', default=False, action="store_true",
                    help="Use arDCA file instead of bmDCA")
    parser.add_argument('--theta', default=0.0, type=float,
                    help="if hamming_dist(msa[i],msa[j])<theta: sum_dist[i] += 1 sum_dist[j] += 1 weight[i] = 1/(sum_dist[i]+1)")
    args = parser.parse_args()
    print(args)
    infer_partner(args.fasta_name, args.graph_name, arDCA=args.arDCA, theta=args.theta)
