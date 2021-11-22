# 2_states_generation_inference

Here, we work with a model infered on a MSA generated with our toy model with two states. See **generating_data_fasta.ipynb**

Two ways to infer a model :

## arDCA

The jupyter notebook **inference_arDCA.ipynb** is used to infer field with **arDCA** on *"Phylogeny_Partners_gene_inf/data_ardca/msa_phylo.npy"* and *"Phylogeny_Partners_gene_inf/data_ardca/msa_no_phylo.npy"*

## bmDCA

You have the choice to use :
    - a modified version of bmDCA in the folder bmdDCA_2_states (It is a modification of [bmdca of matteofigliuzzi](https://github.com/matteofigliuzzi/bmDCA))
    - Or use [bmdca of ranganathanlab]https://github.com/ranganathanlab/bmDCA (I recommend this method)

