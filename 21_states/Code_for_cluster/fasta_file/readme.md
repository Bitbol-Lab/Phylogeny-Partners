# Method

To infer the tree do (you need to install fastree2 before):

```
fasttree Concat_nnn_withFirst.fasta > Concat_nnn_withFirst.fasta.tree
```

After modified the tree to allow a good reproduction of the msa species:

```
python modify_tree.py Concat_nnn_withFirst.fasta.tree Concat_nnn_withFirst_modify.fasta.tree Concat_nnn_withFirst.fasta
```
