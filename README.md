# Phylogeny-Partners

Report : [link overleaf](https://www.overleaf.com/read/hmzwzbgmhwbk)

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

The code works on linux and Mac. (It is possible to adapt it for windows, you need to change the drand48() or random() function by one available in windows distribution)

### Utilisation

You can navigate to figures.ipynb and different_graph.ipynb to see all figures genereted with the simple model.


## Twenty one states models

### Installation 

The code present in Code_for_cluster was present in the EPFL cluster. It useful to genereate data and to infer contact and partners. The inference of the model has been previously done with bmDCA on the cluster, and with CCMgen and arDCA on my personal computer.
If you want to reproduce the data, I advice you to copy the folder Code_for_cluster and to do :

```
cd 21_states/Code_for_cluster/cython_code/
cythonize -i generation_sequence.pyx
cythonize -i generation_sequence_arDCA.pyx
cythonize -i analyse_sequence.pyx 
```

### Generation of data

And after, you can generated MSA (data set of aligned sequences) :

```
sbatch generation_sequence_CCMgen.run
sbatch generation_sequence_bmDCA.run
sbatch generation_sequence_arDCA.run
```

If you are not on a cluster and cannot use the function sbatch, I think that you can replace sbatch by bash. 

### Inference

When these three programmes are completed, you can infer the contact with :

```
sbatch inference_contact.run
```
And for the partners :
```
sbatch inference_partners_generated_data.py
```

Now, if you want to see the plot from the inference you can run the jupyter notebook inside the folder 21_states.


### Note for me :
For pushing new file :
```
git add -A
git commit -m "Name of commit"
git push <remote> <name-of-branch>
git push -u origin new-features
```

