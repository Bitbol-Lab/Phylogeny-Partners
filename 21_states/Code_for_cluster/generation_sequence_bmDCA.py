import numpy as np
import cython_code.generation_sequence as ge
import field_bmDCA.import_msa as im
from Bio import Phylo

tree_file = "fasta_file/Concat_nnn_withFirst_modify.fasta.tree"
tree = Phylo.read(tree_file, "newick")
path_file = "field_bmDCA/parameters_learnt_1980.txt"
Field, Coupling = im.import_msa_bmDCA(path_file)

Flip_equi = 1000000
n_sequences_no_phylo = 23633

msa_gen = ge.Creation_MSA_Generation(Field, Coupling)

msa_no_phylo = msa_gen.msa_no_phylo(n_sequences_no_phylo,Flip_equi)
msa_phylo_tree = msa_gen.msa_tree_phylo(tree.clade, 0)
msa_phylo_equi_tree = msa_gen.msa_tree_phylo(tree.clade, Flip_equi)

np.save("data_bmdca/msa_no_phylo_bmdca", msa_no_phylo)
np.save("data_bmdca/msa_phylo_bmdca_tree", msa_phylo_tree)
np.save("data_bmdca/msa_phylo_equi_bmdca_tree", msa_phylo_equi_tree)
