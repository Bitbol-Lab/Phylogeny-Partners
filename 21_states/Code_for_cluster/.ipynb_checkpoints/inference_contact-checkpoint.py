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

def Inference_contact_real_data(msa, regularisation, n_state_spin, theta):
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
    LTP = [];l_contact=[];L_inter = [];L_intra_hk = [];L_intra_rr = []
    i=-2
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
    return L_inter,L_intra_hk,L_intra_rr,Indice_top_coupling

def percent_list(LTP):
    for pos in range(1,len(LTP)):
        LTP[pos] = (LTP[pos] + LTP[pos-1]*(pos))/(pos+1)
    return LTP


Graph = nx.read_gexf("data/Graph_HK_and_RR_Threshold_8",node_type = int)
regularisation = 0.5
theta = 0.0
n_state_spin = 21

file_fasta = "fasta_file/Concat_nnn_extr_withFirst.fasta"
msa = extr.get_msa_fasta_file(file_fasta)

#### test ######
msa_phylo_equi_tree_ccmpred = extr.get_msa_fasta_file("data_ccmpred/msa.mcmc_phylo_fast_tree_equi_mutation_rate_auto.fas")
msa_phylo_tree_ccmpred = extr.get_msa_fasta_file("data_ccmpred/msa.mcmc_phylo_fast_tree_random_mutation_rate_auto.fas")
L_inter_phylo_tree_CCMgen,L_intra_hk_phylo_tree_CCMgen,L_intra_rr_phylo_tree_CCMgen,_ = Inference_contact_real_data(msa_phylo_tree_ccmpred, regularisation, n_state_spin, theta)
L_inter_phylo_equi_tree_CCMgen,L_intra_hk_phylo_equi_tree_CCMgen,L_intra_rr_phylo_equi_tree_CCMgen,_ = Inference_contact_real_data(msa_phylo_equi_tree_ccmpred, regularisation, n_state_spin, theta)
np.save("output_inference_contact/L_inter_phylo_tree_CCMgen_rate_auto",L_inter_phylo_tree_CCMgen)
np.save("output_inference_contact/L_inter_phylo_equi_tree_CCMgen_rate_auto",L_inter_phylo_equi_tree_CCMgen)
np.save("output_inference_contact/L_intra_hk_phylo_tree_CCMgen_rate_auto",L_intra_hk_phylo_tree_CCMgen)
np.save("output_inference_contact/L_intra_hk_phylo_equi_tree_CCMgen_rate_auto",L_intra_hk_phylo_equi_tree_CCMgen)
np.save("output_inference_contact/L_intra_rr_phylo_tree_CCMgen_rate_auto",L_intra_rr_phylo_tree_CCMgen)
np.save("output_inference_contact/L_intra_rr_phylo_equi_tree_CCMgen_rate_auto",L_intra_rr_phylo_equi_tree_CCMgen)
################
msa_no_phylo_ccmpred = extr.get_msa_fasta_file("data_ccmpred/msa.mcmc_no_phylo.fas")
msa_phylo_tree_ccmpred = extr.get_msa_fasta_file("data_ccmpred/msa.mcmc_phylo_fast_tree_random_mutation_rate_1.fas") 
msa_phylo_equi_tree_ccmpred = extr.get_msa_fasta_file("data_ccmpred/msa.mcmc_phylo_fast_tree_equi_mutation_rate_1.fas")


msa_no_phylo_bmdca = np.load("data_bmdca/msa_no_phylo_bmdca.npy")
msa_phylo_tree_bmdca = np.load("data_bmdca/msa_phylo_bmdca_tree.npy")
msa_phylo_equi_tree_bmdca = np.load("data_bmdca/msa_phylo_equi_bmdca_tree.npy")

msa_no_phylo_ardca = np.load("data_ardca/msa_no_phylo_ardca.npy")
msa_phylo_tree_ardca = np.load("data_ardca/msa_phylo_ardca_tree.npy")
msa_phylo_equi_tree_ardca = np.load("data_ardca/msa_phylo_equi_ardca_tree.npy") 


L_inter_no_phylo_CCMgen,L_intra_hk_no_phylo_CCMgen,L_intra_rr_no_phylo_CCMgen,_ = Inference_contact_real_data(msa_no_phylo_ccmpred, regularisation, n_state_spin, theta)
L_inter_phylo_tree_CCMgen,L_intra_hk_phylo_tree_CCMgen,L_intra_rr_phylo_tree_CCMgen,_ = Inference_contact_real_data(msa_phylo_tree_ccmpred, regularisation, n_state_spin, theta)
L_inter_phylo_equi_tree_CCMgen,L_intra_hk_phylo_equi_tree_CCMgen,L_intra_rr_phylo_equi_tree_CCMgen,_ = Inference_contact_real_data(msa_phylo_equi_tree_ccmpred, regularisation, n_state_spin, theta)

L_inter_no_phylo_bmDCA,L_intra_hk_no_phylo_bmDCA,L_intra_rr_no_phylo_bmDCA,_ = Inference_contact_real_data(msa_no_phylo_bmdca, regularisation, n_state_spin, theta)
L_inter_phylo_bmDCA_tree,L_intra_hk_phylo_bmDCA_tree,L_intra_rr_phylo_bmDCA_tree,_ = Inference_contact_real_data(msa_phylo_tree_bmdca, regularisation, n_state_spin, theta)
L_inter_phylo_equi_bmDCA_tree,L_intra_hk_phylo_equi_bmDCA_tree,L_intra_rr_phylo_equi_bmDCA_tree,_ = Inference_contact_real_data(msa_phylo_equi_tree_bmdca, regularisation, n_state_spin, theta)

L_inter_no_phylo_arDCA,L_intra_hk_no_phylo_arDCA,L_intra_rr_no_phylo_arDCA,_ = Inference_contact_real_data(msa_no_phylo_ardca, regularisation, n_state_spin, theta)
L_inter_phylo_arDCA_tree,L_intra_hk_phylo_arDCA_tree,L_intra_rr_phylo_arDCA_tree,_ = Inference_contact_real_data(msa_phylo_tree_ardca, regularisation, n_state_spin, theta)
L_inter_phylo_equi_arDCA_tree,L_intra_hk_phylo_equi_arDCA_tree,L_intra_rr_phylo_equi_arDCA_tree,_ = Inference_contact_real_data(msa_phylo_equi_tree_ardca, regularisation, n_state_spin, theta)

L_inter,L_intra_hk,L_intra_rr,_ = Inference_contact_real_data(msa, regularisation, n_state_spin, theta)

np.save("output_inference_contact/L_inter",L_inter)
np.save("output_inference_contact/L_inter_no_phylo_CCMgen",L_inter_no_phylo_CCMgen)
np.save("output_inference_contact/L_inter_phylo_tree_CCMgen",L_inter_phylo_tree_CCMgen)
np.save("output_inference_contact/L_inter_phylo_equi_tree_CCMgen",L_inter_phylo_equi_tree_CCMgen)

np.save("output_inference_contact/L_inter_no_phylo_bmDCA",L_inter_no_phylo_bmDCA)
np.save("output_inference_contact/L_inter_phylo_bmDCA_tree",L_inter_phylo_bmDCA_tree)
np.save("output_inference_contact/L_inter_phylo_equi_bmDCA_tree",L_inter_phylo_equi_bmDCA_tree)

np.save("output_inference_contact/L_inter_no_phylo_arDCA",L_inter_no_phylo_arDCA)
np.save("output_inference_contact/L_inter_phylo_arDCA_tree",L_inter_phylo_arDCA_tree)
np.save("output_inference_contact/L_inter_phylo_equi_arDCA_tree",L_inter_phylo_equi_arDCA_tree)


np.save("output_inference_contact/L_intra_hk",L_intra_hk)
np.save("output_inference_contact/L_intra_hk_no_phylo_CCMgen",L_intra_hk_no_phylo_CCMgen)
np.save("output_inference_contact/L_intra_hk_phylo_tree_CCMgen",L_intra_hk_phylo_tree_CCMgen)
np.save("output_inference_contact/L_intra_hk_phylo_equi_tree_CCMgen",L_intra_hk_phylo_equi_tree_CCMgen)

np.save("output_inference_contact/L_intra_hk_no_phylo_bmDCA",L_intra_hk_no_phylo_bmDCA)
np.save("output_inference_contact/L_intra_hk_phylo_bmDCA_tree",L_intra_hk_phylo_bmDCA_tree)
np.save("output_inference_contact/L_intra_hk_phylo_equi_bmDCA_tree",L_intra_hk_phylo_equi_bmDCA_tree)

np.save("output_inference_contact/L_intra_hk_no_phylo_arDCA",L_intra_hk_no_phylo_arDCA)
np.save("output_inference_contact/L_intra_hk_phylo_arDCA_tree",L_intra_hk_phylo_arDCA_tree)
np.save("output_inference_contact/L_intra_hk_phylo_equi_arDCA_tree",L_intra_hk_phylo_equi_arDCA_tree)

np.save("output_inference_contact/L_intra_rr",L_intra_rr)
np.save("output_inference_contact/L_intra_rr_no_phylo_CCMgen",L_intra_rr_no_phylo_CCMgen)
np.save("output_inference_contact/L_intra_rr_phylo_tree_CCMgen",L_intra_rr_phylo_tree_CCMgen)
np.save("output_inference_contact/L_intra_rr_phylo_equi_tree_CCMgen",L_intra_rr_phylo_equi_tree_CCMgen)

np.save("output_inference_contact/L_intra_rr_no_phylo_bmDCA",L_intra_rr_no_phylo_bmDCA)
np.save("output_inference_contact/L_intra_rr_phylo_bmDCA_tree",L_intra_rr_phylo_bmDCA_tree)
np.save("output_inference_contact/L_intra_rr_phylo_equi_bmDCA_tree",L_intra_rr_phylo_equi_bmDCA_tree)

np.save("output_inference_contact/L_intra_rr_no_phylo_arDCA",L_intra_rr_no_phylo_arDCA)
np.save("output_inference_contact/L_intra_rr_phylo_arDCA_tree",L_intra_rr_phylo_arDCA_tree)
np.save("output_inference_contact/L_intra_rr_phylo_equi_arDCA_tree",L_intra_rr_phylo_equi_arDCA_tree)

plt.subplot(1,3,1)
plt.plot(L_inter[:100],color=cmaps(0),label="real data")
plt.plot(L_inter_no_phylo_CCMgen[:100],color=cmaps(1),label="no phylo")
plt.plot(L_inter_phylo_tree_CCMgen[:100],color=cmaps(2),label="phylo tree start random")
plt.plot(L_inter_phylo_equi_tree_CCMgen[:100],color=cmaps(3),label="phylo tree start equi")

plt.plot(L_inter_no_phylo_bmDCA[:100],ls='--',color=cmaps(1),label="no phylo bmDCA")
plt.plot(L_inter_phylo_bmDCA_tree[:100],ls='--',color=cmaps(2),label="phylo start random bmDCA")
plt.plot(L_inter_phylo_equi_bmDCA_tree[:100],ls='--',color=cmaps(3),label="phylo start equi bmDCA")

plt.plot(L_inter_no_phylo_arDCA[:100],ls='-.',color=cmaps(1),label="no phylo arDCA")
plt.plot(L_inter_phylo_arDCA_tree[:100],ls='-.',color=cmaps(2),label="phylo tree start random arDCA")
plt.plot(L_inter_phylo_equi_arDCA_tree[:100],ls='-.',color=cmaps(3),label="phylo tree start equi arDCA")
plt.xlabel("Number of prediction")
plt.ylabel("Proportion of correct prediction")
plt.title("Inter contact")
plt.legend()

plt.subplot(1,3,2)
plt.plot(L_intra_hk[:100],color=cmaps(0),label="real data")
plt.plot(L_intra_hk_no_phylo_CCMgen[:100],color=cmaps(1),label="no phylo")
plt.plot(L_intra_hk_phylo_tree_CCMgen[:100],color=cmaps(2),label="phylo tree start random")
plt.plot(L_intra_hk_phylo_equi_tree_CCMgen[:100],color=cmaps(3),label="phylo tree start equi")

plt.plot(L_intra_hk_no_phylo_bmDCA[:100],ls='--',color=cmaps(1),label="no phylo bmDCA")
plt.plot(L_intra_hk_phylo_bmDCA_tree[:100],ls='--',color=cmaps(2),label="phylo tree start random bmDCA")
plt.plot(L_intra_hk_phylo_equi_bmDCA_tree[:100],ls='--',color=cmaps(3),label="phylo tree start equi bmDCA")

plt.plot(L_intra_hk_no_phylo_arDCA[:100],ls='-.',color=cmaps(1),label="no phylo arDCA")
plt.plot(L_intra_hk_phylo_arDCA_tree[:100],ls='-.',color=cmaps(2),label="phylo tree start random arDCA")
plt.plot(L_intra_hk_phylo_equi_arDCA_tree[:100],ls='-.',color=cmaps(3),label="phylo tree start equi arDCA")

plt.xlabel("Number of prediction")
plt.ylabel("Proportion of correct prediction")
plt.title("Intra HK contact")


plt.subplot(1,3,3)
plt.plot(L_intra_rr[:100],color=cmaps(0),label="real data")
plt.plot(L_intra_rr_no_phylo_CCMgen[:100],color=cmaps(1),label="no phylo")
plt.plot(L_intra_rr_phylo_tree_CCMgen[:100],color=cmaps(2),label="phylo tree start random")
plt.plot(L_intra_rr_phylo_equi_tree_CCMgen[:100],color=cmaps(3),label="phylo tree start equi")

plt.plot(L_intra_rr_no_phylo_bmDCA[:100],ls='--',color=cmaps(1),label="no phylo bmDCA")
plt.plot(L_intra_rr_phylo_bmDCA_tree[:100],ls='--',color=cmaps(2),label="phylo tree start random bmDCA")
plt.plot(L_intra_rr_phylo_equi_bmDCA_tree[:100],ls='--',color=cmaps(3),label="phylo tree start equi bmDCA")

plt.plot(L_intra_rr_no_phylo_arDCA[:100],ls='-.',color=cmaps(1),label="no phylo arDCA")
plt.plot(L_intra_rr_phylo_arDCA_tree[:100],ls='-.',color=cmaps(2),label="phylo tree start random arDCA")
plt.plot(L_intra_rr_phylo_equi_arDCA_tree[:100],ls='-.',color=cmaps(3),label="phylo tree start equi arDCA")
plt.xlabel("Number of prediction")
plt.ylabel("Proportion of correct prediction")
plt.title("Intra RR contact")

plt.suptitle("%s : $\lambda =$ %s, $\\theta = $ %s"%(Graph.name,regularisation,theta))
name = "Inference_contact_%s_lambda=%stheta=%s"%(Graph.name,regularisation,theta) +".png"
plt.savefig("output_inference_contact/"+name)



###### OLD CODE #########
def mean_contact_inference(l_msa, regularisation, n_state_spin, theta):
    l_msa = np.array(l_msa)
    m_inter,m_intra_hk,m_intra_rr = [],[],[]
    for i in range(l_msa.shape[0]):
        msa = l_msa[i]
        print(np.shape(msa))
        L_inter,L_intra_hk,L_intra_rr,_= Inference_contact_real_data(msa, regularisation, n_state_spin, theta)
        m_inter.append(L_inter[0:100])
        m_intra_hk.append(L_intra_hk[0:100])
        m_intra_rr.append(L_intra_rr[0:100])
        print(np.shape(m_inter))
    return np.mean(m_inter,axis=0),np.mean(m_intra_hk,axis=0),np.mean(m_intra_rr,axis=0)
        