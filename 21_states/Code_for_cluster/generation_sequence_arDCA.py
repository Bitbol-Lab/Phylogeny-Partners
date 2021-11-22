import numpy as np
import cython_code.generation_sequence_arDCA as ge
from Bio import Phylo


tree = Phylo.read("fasta_file/Concat_nnn_withFirst_modify.fasta.tree", "newick")
J = np.array(np.load("field_arDCA/J.npy"),order="C")
H = np.array(np.load("field_arDCA/H.npy"),order="C")
p0 = np.array(np.load("field_arDCA/p0.npy"),order="C")
idxperm = np.array(np.load ("field_arDCA/idxperm.npy"),order="C") - 1 ## Convert to python style

msa_gen = ge.Creation_MSA_Generation_arDCA(H,J,p0,idxperm)

msa_phylo = msa_gen.msa_tree_phylo(tree.clade, 0)
msa_phylo_equi = msa_gen.msa_tree_phylo(tree.clade, 1)
msa_no_phylo = msa_gen.msa_no_phylo(msa_phylo.shape[0])

np.save("data_ardca/msa_no_phylo_ardca", msa_no_phylo)
np.save("data_ardca/msa_phylo_ardca_tree", msa_phylo)
np.save("data_ardca/msa_phylo_equi_ardca_tree", msa_phylo_equi)


