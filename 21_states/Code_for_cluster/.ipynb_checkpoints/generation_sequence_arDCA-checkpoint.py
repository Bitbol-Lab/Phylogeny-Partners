import numpy as np
import cython_code.generation_sequence_arDCA as ge
from Bio import Phylo


tree = Phylo.read("fasta_file/Concat_nnn_withFirst.fasta.tree", "newick")
J = np.array(np.load("field_arDCA//J.npy"),order="C")
H = np.array(np.load("field_arDCA//H.npy"),order="C")
p0 = np.array(np.load("field_arDCA//p0.npy"),order="C")
idxperm = np.array(np.load ("field_arDCA/idxperm.npy"),order="C")


n_generations = 14 # 2^10 = 1024 chain of spin
n_mutations_generation = 100
n_avg = 1
n_sequence_no_phylo = 25000

msa_gen = ge.Creation_MSA_Generation_arDCA(H,J,p0,idxperm)
l_msa_no_phylo = []
l_msa_phylo = []
l_msa_phylo_equi = []
l_msa_phylo_tree = []
l_msa_phylo_tree_equi = []

for i in range(n_avg):
    msa_phylo = msa_gen.msa_tree_phylo(tree.clade, 0)
    msa_phylo_equi = msa_gen.msa_tree_phylo(tree.clade, 1)
    l_msa_phylo_tree.append(msa_phylo)
    l_msa_phylo_tree_equi.append(msa_phylo_equi)

    msa_no_phylo = msa_gen.msa_no_phylo(n_sequence_no_phylo)
    msa_phylo = msa_gen.msa_phylo(n_generations, n_mutations_generation, 0)
    msa_phylo_equi = msa_gen.msa_phylo(n_generations, n_mutations_generation, 1)
    l_msa_no_phylo.append(np.copy(msa_no_phylo))
    l_msa_phylo.append(np.copy(msa_phylo))
    l_msa_phylo_equi.append( np.copy(msa_phylo_equi))

np.save("data/l_msa_no_phylo_arDCA", l_msa_no_phylo)
np.save("data/l_msa_phylo_arDCA", l_msa_phylo)
np.save("data/l_msa_phylo_equi_arDCA", l_msa_phylo_equi)

np.save("data/l_msa_phylo_arDCA_tree", l_msa_phylo_tree)
np.save("data/l_msa_phylo_equi_arDCA_tree", l_msa_phylo_tree_equi)


