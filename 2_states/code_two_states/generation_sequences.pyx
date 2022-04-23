# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

import numpy as np
cimport cython
from posix.stdlib cimport random, srandom, drand48
from libc.math cimport exp
from libc.time cimport time
from cython.parallel import prange


cdef class Sampling_msa_two_states:
    cdef int[:,:] m_neighboors
    cdef int[::1] l_stop
    cdef int n_nodes
    cdef public float T 
    cdef public int flip_equi
    
    def __init__(self, graph, int flip_equi = 4000, float T = float(1.0)):
        """
        """
        self.n_nodes = np.intc(graph.number_of_nodes())
        self.T = float(T)
        self.flip_equi = flip_equi
        
        print("Standard Temperature is %s"%self.T)
        print("Number of flip to reach equilibrium is %s"%self.flip_equi)
        
        m_neighboors = np.array(-1*np.ones((graph.number_of_nodes(),graph.number_of_nodes()),dtype=np.intc))
        l_stop = np.zeros((graph.number_of_nodes()),dtype=np.intc)
        for node in range(graph.number_of_nodes()):
            L_neigh = [n for n in graph.neighbors(node)]
            if len(L_neigh)>0:
                m_neighboors[node,:len(L_neigh)] = np.array(L_neigh,dtype=np.intc)
            l_stop[node] = len(L_neigh)
        self.m_neighboors = m_neighboors
        self.l_stop = l_stop
        srandom(<unsigned int>time(NULL))
      
    def msa_no_phylo(self, int n_seq_msa, int wolf_flip = 0):
        """
         int n_seq_msa, int wolf_flip = 0
        """
        cdef:
            int i,j
            char[:,::1] msa = np.array(((np.random.rand(n_seq_msa, self.n_nodes)>0.5)*1.0 - 0.5)*2.0,dtype = np.int8)

        if wolf_flip == 0:
            for i in range(msa.shape[0]):
                self.mcmc(self.flip_equi, msa[i])
        else:
            #print("Use wolf algo with : %s flips"%wolf_flip)
            for i in range(msa.shape[0]):
                for j in range(wolf_flip):
                    self.mcmc_wolf(msa[i])
        return np.asarray(msa)        
                    
    def msa_binary_tree(self, int n_generations, int n_mutations_branch, int start_from_equilibrium, int wolf_flip=0):
        """
         int n_generations, int n_mutations_branch, int start_from_equilibrium, int wolf_flip = 0
        """
        cdef:
            char[::1] seq = np.array(((np.random.rand(self.n_nodes)>0.5)*1.0 - 0.5)*2.0,dtype = np.int8)
            char[:,::1] msa = np.zeros((int(2**n_generations), self.n_nodes),dtype = np.int8)
            int generation,i
            
        if start_from_equilibrium !=0:
            if wolf_flip == 0:
                self.mcmc(self.flip_equi, seq)
            else:
                print("Use wolf algo with : %s flips"%wolf_flip)
                for i in range(wolf_flip):
                    self.mcmc_wolf(seq)
            
        msa[0] = seq

        for generation in range(1,n_generations+1):
            msa[int(2**(generation-1)):int(2**generation),:] = msa[0:int(2**(generation-1)),:]
            for i in range(int(2**generation)):
                self.mcmc(n_mutations_branch, msa[i])
        
        return np.asarray(msa)
            
    def msa_binary_tree_pure_phylo(self, int n_generations, int n_mutations_branch, int start_from_equilibrium, int wolf_flip=0):
        """
         int n_generations, int n_mutations_branch, int start_from_equilibrium, int wolf_flip = 0
        """
        cdef:
            char[::1] seq = np.array(((np.random.rand(self.n_nodes)>0.5)*1.0 - 0.5)*2.0,dtype = np.int8)
            char[:,::1] msa = np.zeros((int(2**n_generations), self.n_nodes),dtype = np.int8)
            int generation,i
            
        if start_from_equilibrium !=0:
            if wolf_flip == 0:
                self.mcmc(self.flip_equi, seq)
            else:
                #print("Use wolf algo with : %s flips"%wolf_flip)
                for i in range(wolf_flip):
                    self.mcmc_wolf(seq)
            
        msa[0] = seq

        for generation in range(1,n_generations+1):
            msa[int(2**(generation-1)):int(2**generation),:] = msa[0:int(2**(generation-1)),:]
            for i in range(int(2**generation)):
                self.mcmc_phylo(n_mutations_branch, msa[i])
        
        return np.asarray(msa)

    cpdef void mcmc(self, int n_mut, char[::1] seq):
        """
        int n_mut, char[::1] seq
        
        Do n_mut mutations on the seq
        """
        cdef:
            int node, c = 0
            float prob
            
        while c<n_mut:   
            node = int(drand48()*self.n_nodes)
            prob = exp(-2.0/(self.T)*self.product_spin(node, seq))
            if prob>1.0 or drand48()<=prob:
                seq[node]= -1*seq[node]
                c += 1
                
    cpdef void mcmc_phylo(self, int n_mut, char[::1] seq):
        """
        int n_mut, char[::1] seq
        
        Do n_mut mutations on the seq
        """
        cdef:
            int node, c = 0
            
        for c in range(n_mut):   
            node = int(drand48()*self.n_nodes)
            seq[node]= -1*seq[node]

    cdef inline float product_spin(self, int node, char[::1] seq) nogil:
        cdef:
            int i
            float result = 0.0
        for i in range(self.l_stop[node]):
            result += seq[self.m_neighboors[node,i]]
        return result*seq[node]
    

    cpdef void mcmc_wolf(self, char[::1] L_spin):
        cdef:
            int[::1] L_node_cluster = -1*np.ones((self.n_nodes),dtype = np.intc)
            int selected_node = np.random.randint(0,self.n_nodes,dtype=np.intc)
            int counter_new_node = 1, counter_old_node = 0, new_node_cluster, node_around_new_node, node_test, save_counter_new_node, save_counter_old_node
            int index_new_node_cluster,index_node_cluster
            float Proba_acceptance_in_cluster_wolf = (1-exp(-2.0/self.T))
        L_node_cluster[0] = selected_node
        while (counter_new_node - counter_old_node)!=0 :
            save_counter_new_node = counter_new_node
            save_counter_old_node = counter_old_node
            counter_old_node = counter_new_node
            for index_new_node_cluster in range(save_counter_old_node,save_counter_new_node):
                new_node_cluster = L_node_cluster[index_new_node_cluster]
                for index_node_around_new_node in range(self.l_stop[new_node_cluster]):
                    node_around_new_node = self.m_neighboors[new_node_cluster,index_node_around_new_node]
                    if L_spin[new_node_cluster]==L_spin[node_around_new_node] and item_in_liste(L_node_cluster[:counter_new_node],node_around_new_node)==0:                    
                        if drand48()<Proba_acceptance_in_cluster_wolf:
                            L_node_cluster[counter_new_node] = node_around_new_node
                            counter_new_node+=1   
        for index_node_cluster in range(counter_new_node):
            L_spin[L_node_cluster[index_node_cluster]] = -1*L_spin[L_node_cluster[index_node_cluster]]
            

cdef int item_in_liste(int[::1] List, int item) nogil:
    cdef int item_liste, index_item_liste
    for index_item_liste in range(len(List)):
        item_liste = List[index_item_liste]
        if item_liste == item:
            return 1
        if item_liste == -1:
            break
    return 0
    

def all_hamming_dist(char[:,::1] msa):
    cdef:
        double[:,::1] dist = np.ones((msa.shape[0],msa.shape[0]))
        int i,j
    with nogil:
        for i in prange(msa.shape[0]-1):
            for j in range(i+1,msa.shape[0]):
                dist[i,j] = hamming_dist(msa[i],msa[j])
    return np.asarray(dist)

cpdef double[::1] weight_msa(char[:,::1] msa, double theta):
    cdef:
        double[::1] weight = np.ones((msa.shape[0]))
        double[::1] sum_dist = np.zeros((msa.shape[0]))
        int i,j
    if theta>0:
        with nogil:
            for i in prange(msa.shape[0]-1):
                for j in range(i+1,msa.shape[0]):
                    if hamming_dist(msa[i],msa[j])<theta:
                        sum_dist[i] += 1
                        sum_dist[j] += 1
    for i in range(msa.shape[0]):
        weight[i] = 1/(sum_dist[i]+1) #dont take into account the diagonal
    return np.asarray(weight)

cdef double hamming_dist(char[::1] a, char[::1] b) nogil:
    cdef:
        int i
        double dist = 0.0
    for i in range(a.shape[0]):
        dist+= (a[i]!=b[i])*1.0
    return dist/a.shape[0]