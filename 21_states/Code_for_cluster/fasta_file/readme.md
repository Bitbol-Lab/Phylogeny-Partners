# Method

To infer the tree do (you need to install fastree2 before):

```
fasttree Concat_nnn_withFirst.fasta > Concat_nnn_withFirst.fasta.tree
```

After modified the tree to allow a good reproduction of the msa species:

```
python modify_tree.py Concat_nnn_withFirst.fasta.tree Concat_nnn_withFirst_modify.fasta.tree Concat_nnn_withFirst.fasta
```
Ou pour MALG_MALK :

```
python modify_tree.py MALG_MALK_cov75_hmmsearch_sorted_withLast_b.fas.tree MALG_MALK_cov75_hmmsearch_sorted_withLast_b_modify.fas.tree MALG_MALK_cov75_hmmsearch_sorted_withLast_b.fas
```

## To observe the tree inferred 

figtree Concat_nnn_withFirst.fasta.tree 
