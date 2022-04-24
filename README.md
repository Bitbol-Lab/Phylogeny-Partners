# Phylogeny-Partners

[![DOI](https://zenodo.org/badge/430666146.svg)](https://zenodo.org/badge/latestdoi/430666146)

## Two states models

### Instalation
  You may need to install the cython, networkx, numpy, scipy package:
  ```
  pip install cython, networkx, numpy, scipy
  ```
  and do in the folder two_states :
  ```
  cd 2_states/code_two_states
  cythonize -i generation_sequences.pyx
  cythonize -i mutual_info.pyx
  ```
To use the last part of the jupyter notebook **figures.ipynb**, you should install bmDCA :
```
https://github.com/ranganathanlab/bmDCA
```


The code works on linux and Mac. (It is possible to adapt it for windows, you need to change the drand48() or random() function by one available in windows distribution)

### Utilisation

You can open **figures.ipynb** and **different_graph.ipynb** to see all figures generated with the simple model.


## Twenty one states models

### Installation 

The code present in Code_for_cluster was present in the EPFL cluster. It is helpful to generate data and to infer contact and partners. The model has been inferred with bmDCA on the cluster and arDCA on my personal computer.
If you want to reproduce the data, I advise you to copy the folder Code_for_cluster and to do :

```
cd 21_states/Code_for_cluster/cython_code/
cythonize -i generation_sequence.pyx
cythonize -i generation_sequence_arDCA.pyx
cythonize -i analyse_sequence.pyx 
```

### Generation of data

And after, you can generate MSA (data set of aligned sequences) :

```
sbatch generation_sequence_bmDCA.run
sbatch generation_sequence_arDCA.run
```

If you are not on a cluster and cannot use the function sbatch, you can replace sbatch by bash. 

### Inference

When these three programmes are completed, you can infer the contact with :

```
sbatch inference_contact.run
```
And for the partners :
```
sbatch inference_partners_generated_data.py
```

If you want to see the plot from the inference, you can run the jupyter notebook inside the folder 21_states.

## 2 states with sampling on a inferred model

### Installation 

See **2_states_generation_inference/README.md**

### Figures
See **generating_data_fasta.ipynb**

