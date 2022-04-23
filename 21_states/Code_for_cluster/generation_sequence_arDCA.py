import numpy as np
import cython_code.generation_sequence_arDCA as ge
from Bio import Phylo
import sys

def hk_rr():
    ## HK RR
    name_fasta = "Concat_nnn_withFirst.fasta"
    tree = Phylo.read("fasta_file/Concat_nnn_withFirst_modify.fasta.tree", "newick")
    tree.root_at_midpoint()

    J = np.array(np.load("field_arDCA/data/J_%s.npy"%name_fasta),order="C")
    H = np.array(np.load("field_arDCA/data/H_%s.npy"%name_fasta),order="C")
    p0 = np.array(np.load("field_arDCA/data/p0_%s.npy"%name_fasta),order="C")
    idxperm = np.array(np.load ("field_arDCA/data/idxperm_%s.npy"%name_fasta),order="C") - 1 ## Convert to python style

    msa_gen = ge.Creation_MSA_Generation_arDCA(H,J,p0,idxperm)

    msa_phylo = msa_gen.msa_tree_phylo(tree.clade, 0)
    msa_phylo_equi = msa_gen.msa_tree_phylo(tree.clade, 1)
    msa_no_phylo = msa_gen.msa_no_phylo(msa_phylo.shape[0])

    np.save("data_ardca/msa_no_phylo_ardca_%s"%name_fasta, msa_no_phylo)
    np.save("data_ardca/msa_phylo_ardca_tree_%s"%name_fasta, msa_phylo)
    np.save("data_ardca/msa_phylo_equi_ardca_tree_%s"%name_fasta, msa_phylo_equi)

def malg_malk():
    ## MALG MALK
    name_fasta = "MALG_MALK_cov75_hmmsearch_sorted_withLast_b.fas"
    tree = Phylo.read("fasta_file/MALG_MALK_cov75_hmmsearch_sorted_withLast_b_modify.fas.tree", "newick")
    tree.root_at_midpoint()

    J = np.array(np.load("field_arDCA/data/J_%s.npy"%name_fasta),order="C")
    H = np.array(np.load("field_arDCA/data/H_%s.npy"%name_fasta),order="C")
    p0 = np.array(np.load("field_arDCA/data/p0_%s.npy"%name_fasta),order="C")
    idxperm = np.array(np.load ("field_arDCA/data/idxperm_%s.npy"%name_fasta),order="C") - 1 ## Convert to python style

    msa_gen = ge.Creation_MSA_Generation_arDCA(H,J,p0,idxperm)

    msa_phylo = msa_gen.msa_tree_phylo(tree.clade, 0)
    msa_phylo_equi = msa_gen.msa_tree_phylo(tree.clade, 1)
    msa_no_phylo = msa_gen.msa_no_phylo(msa_phylo.shape[0])

    np.save("data_ardca/msa_no_phylo_ardca_%s"%name_fasta, msa_no_phylo)
    np.save("data_ardca/msa_phylo_ardca_tree_%s"%name_fasta, msa_phylo)
    np.save("data_ardca/msa_phylo_equi_ardca_tree_%s"%name_fasta, msa_phylo_equi)


if __name__ == '__main__':
    print(sys.argv)
    if sys.argv[1] == "0":
        print("arDCA HK-RR")
        hk_rr()
    elif sys.argv[1] == "1":
        print("arDCA malg_malk")
        malg_malk()
    else:
        print("Didn't understand the arg %s"%sys.argv)

