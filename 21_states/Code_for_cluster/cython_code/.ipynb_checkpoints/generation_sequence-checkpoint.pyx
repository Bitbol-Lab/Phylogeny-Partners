# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

import numpy as np
cimport cython
from libc.stdlib cimport RAND_MAX
from posix.stdlib cimport random, srandom, drand48
from libc.math cimport exp
from libc.time cimport time
from cython.parallel import prange

cdef class Creation_MSA_Generation:
    """
    Spin = index of the possible state (i.e classic spin +-1 --> 0 or 1)
    """
    cdef double[:,::1] Field
    cdef double[:,:,:,::1] Coupling
    cdef int Number_state_spin
    cdef int Number_of_Node
    
    def __init__(self, double[:,::1] Field, double[:,:,:,::1] Coupling ):
        """
        int Number_Generation, int Numb_Mutation_Per_Generation, int Number_for_average,
         double[:,::1] Field, double[:,:,:,::1] Coupling,
        int Flip_equi , double threshold_neighboors  
        """
        self.Number_of_Node = np.intc(Field.shape[0])
        self.Number_state_spin = np.intc(Field.shape[1])
        self.Field = Field
        self.Coupling = Coupling
        srandom(<unsigned int>time(NULL))
        
    def msa_no_phylo(self, int n_sequences, int n_flip_equi):
        cdef:
            int index_msa
            char[:,::1] msa = np.random.randint(0,high=self.Number_state_spin, 
                                size = (n_sequences,self.Number_of_Node)
                                ,dtype = np.int8)
        for index_msa in prange(msa.shape[0],nogil=True,schedule='dynamic'):
            self.mcmc(n_flip_equi, msa[index_msa])
        return np.asarray(msa)
                
    def msa_phylo(self, int n_generations, int n_mutations_generation, int flip_before_start):
        cdef:
            char[::1] l_spin = np.random.randint(0,
                                                 high=self.Number_state_spin
                                                 ,size = ( self.Number_of_Node)
                                                ,dtype = np.int8)
            char[:,::1] msa = np.zeros((int(2**n_generations),self.Number_of_Node)
                                ,dtype = np.int8)
            int generation,index_sequence
        self.mcmc(flip_before_start, l_spin)     
        msa[0] = l_spin
        for generation in range(1,n_generations+1):
            msa[int(2**(generation-1)):int(2**generation),:] = msa[0:int(2**(generation-1)),:]
            for index_sequence in range(int(2**generation)):
                self.mcmc(n_mutations_generation, msa[index_sequence])
        return np.asarray(msa)
    
    def msa_tree_phylo(self, clade_root, int flip_before_start, double neff = 1.0):
        cdef :
            char[::1] first_sequence = np.random.randint(0,high=self.Number_state_spin
                                                         ,size = ( self.Number_of_Node)
                                                         ,dtype = np.int8)  
            #char[:,::1] msa = np.random.randint(0,high=self.Number_state_spin, 
            #                                    size = (len(clade_root.get_terminals()),self.Number_of_Node)
            #                                    ,dtype = np.int8)
            char[:,::1] msa = np.zeros( (len(clade_root.get_terminals()),self.Number_of_Node), dtype=np.int8)
        self.mcmc(flip_before_start, first_sequence)
        return np.asarray(self.msa_tree_phylo_recur(clade_root, first_sequence, msa, neff))
    
    cdef char[:,::1] msa_tree_phylo_recur(self, clade_root, char[::1] previous_sequence, char[:,::1] msa, double neff):
        cdef:
            char[::1] new_sequence = np.zeros((previous_sequence.shape[0]),dtype=np.int8)
            int n_mutations
        b = clade_root.clades
        if len(b)>0:
            for clade in b:
                #Mutation on previous_sequences
                new_sequence[:] = previous_sequence
                n_mutations = int(clade.branch_length*new_sequence.shape[0]*neff)
                self.mcmc(n_mutations, new_sequence)
                self.msa_tree_phylo_recur(clade, new_sequence, msa, neff)
        else:
            n_mutations = int(clade_root.branch_length*previous_sequence.shape[0]*neff)
            self.mcmc(n_mutations, previous_sequence)
            msa[int(clade_root.name),:] = previous_sequence
        return msa
    
    def hamiltonian(self,char[::1] L_Spin):
        cdef:
            int node_i,index_neighboor
            double hamiltonian = 0.0
        for node_i in range(self.Number_of_Node-1):
            hamiltonian -= self.Field[node_i,L_Spin[node_i]]
            for index_neighboor in range(node_i+1,self.Number_of_Node):
                hamiltonian -= self.Coupling[node_i,index_neighboor,L_Spin[node_i],L_Spin[index_neighboor]]
        return hamiltonian 
  
    cdef inline void mcmc(self, int Number_of_Mutation, char[::1] L_Spin) nogil:  
        cdef:
            int selected_node, new_state, c_mutation = 0
            double Prob, de
        while c_mutation<Number_of_Mutation:   
            selected_node = randint(0,self.Number_of_Node)
            new_state = randint(0,self.Number_state_spin-1)
            if new_state >= L_Spin[selected_node]:
                new_state += 1 
            de = (
                self.Pseudo_Hamiltonian(selected_node, new_state, L_Spin) -
                self.Pseudo_Hamiltonian(selected_node, L_Spin[selected_node], L_Spin)
                 )
            if de>=0 or drand48()<exp(de):
                L_Spin[selected_node]= new_state
                c_mutation += 1

    cdef inline double Pseudo_Hamiltonian(self, int node, int state_node, char[::1] L_Spin) nogil:
        cdef:
            int i
            double hamiltonian = self.Field[node,state_node] - self.Coupling[node,node,state_node,L_Spin[node]]
        for i in range(L_Spin.shape[0]):
            hamiltonian += self.Coupling[node,i,state_node,L_Spin[i]]
        return hamiltonian 

cdef inline int randint(int lower, int upper) nogil:
    """lower included, upper excluded"""
    return random() % (upper - lower ) + lower 

