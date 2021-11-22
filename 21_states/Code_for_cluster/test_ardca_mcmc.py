import numpy as np
import cython_code.generation_sequence_arDCA as ge
from Bio import Phylo

tree = Phylo.read("fasta_file/Concat_nnn_withFirst.fasta.tree", "newick")
J = np.array(np.load("field_arDCA//J.npy"),order="C")
H = np.array(np.load("field_arDCA//H.npy"),order="C")
p0 = np.array(np.load("field_arDCA//p0.npy"),order="C")
idxperm = np.array(np.load ("field_arDCA/idxperm.npy"),order="C")

msa_gen = ge.Creation_MSA_Generation_arDCA(H,J,p0,idxperm)

n_sequence = 1000
msa_no_phylo_arDCA = msa_gen.msa_no_phylo(n_sequence)
np.save("data_ardca/msa_no_phylo_arDCA", msa_no_phylo_arDCA)

l_flips =[1000, 10000, 10000, 100000, 10000000]

for n_flips in l_flips:
    msa_no_phylo_mcmc = msa_gen.msa_no_phylo_mcmc(n_sequence,n_flips)
    np.save("data_ardca/msa_no_phylo_mcmc_flips_%s"%n_flips, msa_no_phylo_mcmc)
