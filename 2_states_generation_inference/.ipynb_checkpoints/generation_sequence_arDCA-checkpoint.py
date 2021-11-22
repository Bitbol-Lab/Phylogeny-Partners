import numpy as np
import cython_code.generation_sequence_arDCA as ge
from Bio import Phylo


def generate_sequences_arDCA(path_folder_coupling, name_msa, path_save_msa, n_sequences_no_phylo):

    J = np.array(np.load( path_folder_coupling + "J_" + name_msa + ".npy"), order="C")
    H = np.array(np.load( path_folder_coupling + "H_" + name_msa + ".npy"), order="C")
    p0 = np.array(np.load( path_folder_coupling + "p0_" + name_msa + ".npy"), order="C")
    idxperm = np.array(np.load ( path_folder_coupling +  "idxperm_" + name_msa + ".npy"), order="C") - 1 ## Convert to python style

    msa_gen = ge.Creation_MSA_Generation_arDCA(H,J,p0,idxperm)
    
    msa_no_phylo = msa_gen.msa_no_phylo( n_sequences_no_phylo )

    np.save(path_save_msa, msa_no_phylo)


def generate_sequences_arDCA_phylo(path_folder_coupling, name_msa, path_save_msa, n_generations, n_mutations_generation, start_equi):

    J = np.array(np.load( path_folder_coupling + "J_" + name_msa + ".npy"), order="C")
    H = np.array(np.load( path_folder_coupling + "H_" + name_msa + ".npy"), order="C")
    p0 = np.array(np.load( path_folder_coupling + "p0_" + name_msa + ".npy"), order="C")
    idxperm = np.array(np.load ( path_folder_coupling +  "idxperm_" + name_msa + ".npy"), order="C") - 1 ## Convert to python style

    msa_gen = ge.Creation_MSA_Generation_arDCA(H,J,p0,idxperm)
    
    msa_phylo = msa_gen.msa_phylo(n_generations, n_mutations_generation, start_equi)

    np.save(path_save_msa, msa_phylo)
