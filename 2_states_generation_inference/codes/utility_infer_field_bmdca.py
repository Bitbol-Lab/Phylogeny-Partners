import numpy as np
import random_fasta as rdf
import subprocess

def Ising_gauge_Jij(Jij):
    """
    In : Jij L*(q)
    Out : Jij L*q gauge Ising
    """
 
    q = Jij.shape[3]
    L = Jij.shape[0]
    Jij_out = np.zeros((L, q, L, q))

    for index_i in range(L):
        for index_j in range(L):
            mean_two_sites = np.mean(Jij[index_i,index_j])
            for state_1 in range(q):
                mean_second_site = np.mean(Jij[index_i,index_j,state_1])
                for state_2 in range(q):
                    mean_first_site = np.mean(Jij[index_i,index_j,:,state_2])
                    Jij_out[index_i,state_1,index_j,state_2] = Jij[index_i,index_j,state_1,state_2] - mean_first_site - mean_second_site + mean_two_sites
    return Jij_out


def import_msa_bmDCA(path_file):  
    with open(path_file, "r") as f:
        L = 0
        n = 0
        for line in f:
            l = line.rstrip("\n").split(" ")
            val = float(l[-1]) 
            if l[0] == "h":
                if int(l[1])> L:
                    L = int(l[1])
                if int(l[2]) > n:
                    n = int(l[2])
    L += 1
    n += 1
    print("Proteins of %s amino acids and %s states"%(L,n))
    
    J2 = np.zeros((L, L, n, n), dtype=np.float64)
    h = np.zeros((L, n), dtype=np.float64)
    
    with open(path_file, "r") as f:
        for line in f:
            l = line.rstrip("\n").split(" ")
            val = float(l[-1]) 
            if l[0] == "J":
                J2[int(l[1]), int(l[2]), int(l[3]), int(l[4])] = val
            elif l[0] == "h":
                h[int(l[1]), int(l[2])] = val
    # Symmetrize J2
    for i in range(J2.shape[0]):
        for j in range(i):
            J2[i, j, ...] = J2[j, i, ...].T
            
    return h,J2

def infer_bmDCA_fields(
    l_msa : list,
    l_n_mutations_branch : list,
    size_train : int = None,
    return_path_coupling : bool = False
):
    
    assert len(l_msa) == len(l_n_mutations_branch)
    
    folder = "data_modif_tree_mu/"
    config = "bmdca_new/bmdca.conf"
    alphabet = "-A" #First two letter of alphabet used in bmDCA

    l_Jij, l_msa_test, l_path_coupling = [], [], []
    
    for i, msa in enumerate(l_msa):
        
        l_perm = np.random.permutation(msa.shape[0])
        if size_train is None:
            msa_train = msa
            msa_test = None
        else:
            msa_train = msa[l_perm[:size_train]]
            msa_test = msa[l_perm[size_train:]] 
        
        name = "msa_%s_mutations_%s_size_train.fasta"%(l_n_mutations_branch[i], size_train)
        file_name = folder + name
        rdf.write_msa_to_fasta(msa_train, file_name, alphabet)
        
        print(file_name)
        
        directory_result = file_name[:-6] + "_calcul/"
        cmd = "bmdca -i %s -c %s -d %s -r"%(file_name, config, directory_result)
        normal =  subprocess.run(cmd, capture_output=True, shell=True)
        print(normal)
        
        cmd = "arma2ascii -p %s -P %s"%(directory_result + "parameters_h_final.bin",  directory_result + "parameters_J_final.bin")
        normal =  subprocess.run(cmd, capture_output=True, shell=True)
        print(normal)
    
        l_path_coupling.append(directory_result + "parameters_final.txt")
        _, Jij = import_msa_bmDCA(directory_result + "parameters_final.txt")
        
        if return_path_coupling is False:
            Jij = np.array(Jij, order="F", dtype=np.double, subok=True)
            Jij = Ising_gauge_Jij(Jij)

            l_Jij.append(Jij)
            l_msa_test.append(msa_test)
        
    if return_path_coupling is False:
        return l_Jij, l_msa_test
    else:
        return l_path_coupling
