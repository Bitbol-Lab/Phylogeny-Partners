import networkx as nx
from scipy import linalg
import cython_code.analyse_sequence as an
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import fasta_file.extract_msa as extr
cmaps = plt.get_cmap("tab10")
plt.rcParams["figure.figsize"] = (15,5)

def Inference_contact_real_data(msa, regularisation, n_state_spin, theta, Graph):
    msa = np.array(msa)
    L = msa.shape[1]
    n_states_gauge = n_state_spin-1
    Cij_shape4,_ = an.Cij_cython(msa, regularisation, n_state_spin, theta)
    Cij = np.reshape(Cij_shape4,(L*(n_state_spin-1),L*(n_state_spin-1)),order = "F") # F contigunous array for speed up the inverse function wrote in fortran
    Cij = linalg.inv(Cij)
    Jij =  np.reshape(-1*Cij,(L,n_state_spin-1,L,n_state_spin-1),order = "F")
    Jij = an.Ising_gauge_Jij(Jij) #C contiguous array now  !!!
    ### 
    Fij = np.linalg.norm(Jij,axis=(1,3))
    Fi = np.mean(Fij,axis=1)
    F = np.mean(Fij)
    Fij_apc = Fij - np.outer(Fi,Fi)/F
    Indice_top_coupling = np.unravel_index(np.argsort(-Fij_apc, axis=None), Fij.shape)
    LTP = [];l_contact = [];L_inter = [];L_intra_hk = [];L_intra_rr = []
    i = -2
    while len(L_inter)<100 or len(L_intra_hk)<100 or len(L_intra_rr)<100  :
        i+= 2
        i1 = Indice_top_coupling[0][i]
        i2 = Indice_top_coupling[1][i]
        if abs(i1-i2)>4:
            if Graph.nodes[i1]["subset"]!=Graph.nodes[i2]["subset"]:
                L_inter.append((i1 in list(Graph[i2]))*1)
            elif Graph.nodes[i1]["subset"]==1:
                L_intra_hk.append((i1 in list(Graph[i2]))*1)
            else:
                L_intra_rr.append((i1 in list(Graph[i2]))*1)
    L_inter = percent_list(L_inter)
    L_intra_hk = percent_list(L_intra_hk)
    L_intra_rr = percent_list(L_intra_rr)
    return L_inter, L_intra_hk, L_intra_rr, Indice_top_coupling

def percent_list(LTP):
    for pos in range(1,len(LTP)):
        LTP[pos] = (LTP[pos] + LTP[pos-1]*(pos))/(pos+1)
    return LTP



def infer_contact(name_graph, name_fasta, infer_bmdca=True, infer_ardca=True):
    graph = nx.read_gexf("Extract_contact_pdb_HK-RR/%s"%name_graph, node_type = int)
    regularisation = 0.5
    theta = 0.0
    n_state_spin = 21
    file_fasta = "fasta_file/%s"%name_fasta
    msa = extr.get_msa_fasta_file(file_fasta)
    args = [regularisation, n_state_spin, theta, graph]
    L_inter,L_intra_hk,L_intra_rr,_ = Inference_contact_real_data(msa, *args)
    np.save("output_inference_contact/L_inter_%s"%name_graph, L_inter)
    np.save("output_inference_contact/L_intra_hk_%s"%name_graph, L_intra_hk)
    np.save("output_inference_contact/L_intra_rr_%s"%name_graph, L_intra_rr)

    if infer_bmdca:
        msa_no_phylo_bmdca = np.load("data_bmdca/msa_no_phylo_bmdca_%s.npy"%name_fasta)
        msa_phylo_tree_bmdca = np.load("data_bmdca/msa_phylo_bmdca_tree_%s.npy"%name_fasta)
        msa_phylo_equi_tree_bmdca = np.load("data_bmdca/msa_phylo_equi_bmdca_tree_%s.npy"%name_fasta)

        L_inter_no_phylo_bmDCA,L_intra_hk_no_phylo_bmDCA,L_intra_rr_no_phylo_bmDCA,_ = Inference_contact_real_data(msa_no_phylo_bmdca, *args)
        L_inter_phylo_bmDCA_tree,L_intra_hk_phylo_bmDCA_tree,L_intra_rr_phylo_bmDCA_tree,_ = Inference_contact_real_data(msa_phylo_tree_bmdca, *args)
        L_inter_phylo_equi_bmDCA_tree,L_intra_hk_phylo_equi_bmDCA_tree,L_intra_rr_phylo_equi_bmDCA_tree,_ = Inference_contact_real_data(msa_phylo_equi_tree_bmdca, *args)

        np.save("output_inference_contact/L_inter_no_phylo_bmDCA_%s"%name_graph,L_inter_no_phylo_bmDCA)
        np.save("output_inference_contact/L_inter_phylo_bmDCA_tree_%s"%name_graph,L_inter_phylo_bmDCA_tree)
        np.save("output_inference_contact/L_inter_phylo_equi_bmDCA_tree_%s"%name_graph,L_inter_phylo_equi_bmDCA_tree)

        np.save("output_inference_contact/L_intra_hk_no_phylo_bmDCA_%s"%name_graph,L_intra_hk_no_phylo_bmDCA)
        np.save("output_inference_contact/L_intra_hk_phylo_bmDCA_tree_%s"%name_graph,L_intra_hk_phylo_bmDCA_tree)
        np.save("output_inference_contact/L_intra_hk_phylo_equi_bmDCA_tree_%s"%name_graph,L_intra_hk_phylo_equi_bmDCA_tree)

        np.save("output_inference_contact/L_intra_rr_no_phylo_bmDCA_%s"%name_graph,L_intra_rr_no_phylo_bmDCA)
        np.save("output_inference_contact/L_intra_rr_phylo_bmDCA_tree_%s"%name_graph,L_intra_rr_phylo_bmDCA_tree)
        np.save("output_inference_contact/L_intra_rr_phylo_equi_bmDCA_tree_%s"%name_graph,L_intra_rr_phylo_equi_bmDCA_tree) 

    if infer_ardca:
        msa_no_phylo_ardca = np.load("data_ardca/msa_no_phylo_ardca_%s.npy"%name_fasta)
        msa_phylo_tree_ardca = np.load("data_ardca/msa_phylo_ardca_tree_%s.npy"%name_fasta)
        msa_phylo_equi_tree_ardca = np.load("data_ardca/msa_phylo_equi_ardca_tree_%s.npy"%name_fasta) 

        L_inter_no_phylo_arDCA,L_intra_hk_no_phylo_arDCA,L_intra_rr_no_phylo_arDCA,_ = Inference_contact_real_data(msa_no_phylo_ardca, *args)
        L_inter_phylo_arDCA_tree,L_intra_hk_phylo_arDCA_tree,L_intra_rr_phylo_arDCA_tree,_ = Inference_contact_real_data(msa_phylo_tree_ardca, *args)
        L_inter_phylo_equi_arDCA_tree,L_intra_hk_phylo_equi_arDCA_tree,L_intra_rr_phylo_equi_arDCA_tree,_ = Inference_contact_real_data(msa_phylo_equi_tree_ardca, *args)

        np.save("output_inference_contact/L_inter_no_phylo_arDCA_%s"%name_graph,L_inter_no_phylo_arDCA)
        np.save("output_inference_contact/L_inter_phylo_arDCA_tree_%s"%name_graph,L_inter_phylo_arDCA_tree)
        np.save("output_inference_contact/L_inter_phylo_equi_arDCA_tree_%s"%name_graph,L_inter_phylo_equi_arDCA_tree)

        np.save("output_inference_contact/L_intra_hk_no_phylo_arDCA_%s"%name_graph,L_intra_hk_no_phylo_arDCA)
        np.save("output_inference_contact/L_intra_hk_phylo_arDCA_tree_%s"%name_graph,L_intra_hk_phylo_arDCA_tree)
        np.save("output_inference_contact/L_intra_hk_phylo_equi_arDCA_tree_%s"%name_graph,L_intra_hk_phylo_equi_arDCA_tree)

        np.save("output_inference_contact/L_intra_rr_no_phylo_arDCA_%s"%name_graph,L_intra_rr_no_phylo_arDCA)
        np.save("output_inference_contact/L_intra_rr_phylo_arDCA_tree_%s"%name_graph,L_intra_rr_phylo_arDCA_tree)
        np.save("output_inference_contact/L_intra_rr_phylo_equi_arDCA_tree_%s"%name_graph,L_intra_rr_phylo_equi_arDCA_tree)


name_fasta = "MALG_MALK_cov75_hmmsearch_sorted_withLast_b.fas"
name_graph = "prot_MalG_MalK_Threshold_8_MinAllDist"
infer_contact(name_graph, name_fasta, infer_bmdca=False)
name_graph = "prot_MalG_MalK_Threshold_4_MinAllDist"
infer_contact(name_graph, name_fasta,infer_bmdca=False)
name_graph = "prot_MalG_MalK_Threshold_8_CarbonAlpha"
infer_contact(name_graph, name_fasta, infer_bmdca=False)

name_fasta = "Concat_nnn_withFirst.fasta"
name_graph = "prot_HK_and_RR_Threshold_4_MinAllDist"
infer_contact(name_graph, name_fasta)
name_graph = "prot_HK_and_RR_Threshold_8_MinAllDist"
infer_contact(name_graph, name_fasta)
name_graph = "prot_HK_and_RR_Threshold_8_CarbonAlpha"
infer_contact(name_graph, name_fasta)