import numpy as np
import cython_code.generation_sequence as ge
import utilis_bmDCA as im
from Bio import Phylo

def generate_sequences_bmDCA(path_file_coupling, path_save_msa, Flip_equi, n_sequences_no_phylo):

    Field, Coupling = im.import_msa_bmDCA(path_file_coupling)

    msa_gen = ge.Creation_MSA_Generation(Field, Coupling)

    msa_no_phylo = msa_gen.msa_no_phylo(n_sequences_no_phylo, Flip_equi)

    np.save(path_save_msa, msa_no_phylo)
    
    return msa_no_phylo
    
def generate_sequences_bmDCA_phylo(path_file_coupling, path_save_msa, n_generations, n_mutations_generation, flip_before_start):

    Field, Coupling = im.import_msa_bmDCA(path_file_coupling)

    msa_gen = ge.Creation_MSA_Generation(Field, Coupling)

    msa_phylo = msa_gen.msa_phylo(n_generations, n_mutations_generation, flip_before_start)

    np.save(path_save_msa, msa_phylo)
    
    return msa_phylo