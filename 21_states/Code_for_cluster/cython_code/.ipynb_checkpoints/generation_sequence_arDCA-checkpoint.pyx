# cython: language_level=3,  boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

import numpy as np
cimport cython
from libc.stdlib cimport RAND_MAX, rand
from posix.stdlib cimport random, srandom, drand48
from libc.math cimport exp
from libc.time cimport time
from cython.parallel import prange
from libc.time cimport time


cdef class Creation_MSA_Generation_arDCA:
    """
    np.float64_[:,::1] Field,
                 np.float64_[:,:,:,::1] Coupling,
                 np.float64_[::1] p0,
                 long[::1] idxperm
    """
    cdef double[:,::1] Field
    cdef double[:,:,:,::1] Coupling
    cdef double[::1] p0
    cdef long[::1] idxperm
    cdef public int Number_state_spin
    cdef public int Number_of_Node
    cdef public double Temperature
    
    def __init__(self, Field,
                 Coupling,
                 p0,
                 idxperm):
        """
        """
        self.Number_of_Node = np.intc(Field.shape[0])+1
        self.Number_state_spin = np.intc(Field.shape[1])
        self.p0 = p0
        self.idxperm = idxperm ## Need to Convert to Python Style !!!!
        self.Field = Field
        self.Coupling = Coupling
        self.Temperature = 1
        srandom(<unsigned int>time(NULL))
        
    def msa_no_phylo(self, long n_sequences):
        cdef:
            int i,site,j,a,q,N
            char[::1] sample_z
            int[::1] backorder
            char[:,::1] msa
            double[::1] totH,p
        srandom(<unsigned int>time(NULL))
        q = self.p0.shape[0]
        N = self.Field.shape[0]
        msa = np.zeros((n_sequences,N+1),dtype=np.int8)
        backorder = np.array(np.argsort(np.asarray(self.idxperm)),dtype=np.intc)
        totH = np.zeros((q),dtype=np.double)
        sample_z = np.zeros((N+1),dtype=np.int8)
        for i in range(n_sequences):
            sample_z[<int>0] = rand_choice(self.p0)
            for site in range(N):
                totH[:] = self.Field[site]
                for j in range(site+1):
                    for a in range(q):
                        totH[a] += self.Coupling[site,a,sample_z[j],j]
                p = softmax(totH)
                sample_z[site + 1] = rand_choice(p)
            for j in range(sample_z.shape[0]):
                msa[i,j] = sample_z[backorder[j]]
        return np.asarray(msa)
    

    def msa_no_phylo_mcmc(self, int n_sequences, int n_flip_equi):
        cdef:
            int index_msa
            char[:,::1] msa = np.random.randint(0,high=self.Number_state_spin, 
                                size = (n_sequences,self.Number_of_Node)
                                ,dtype = np.int8)
            double[:,:,::1] tot_tottotH = np.zeros((n_sequences,self.Field.shape[0],self.Field.shape[1]))
        for index_msa in prange(msa.shape[0],nogil=True):
            self.mcmc(n_flip_equi, msa[index_msa], tot_tottotH[index_msa] )
        return np.asarray(msa)
                
                
    def msa_phylo(self, int n_generations, int n_mutations_generation, int start_equi):
        """
        int n_generations, int n_mutations_generation, int start_equi
        """
        cdef:
            char[::1] sequence 
            char[:,::1] msa = np.zeros((int(2**n_generations),self.Number_of_Node)
                                ,dtype = np.int8)
            double[:,::1] tottotH = np.zeros((self.Field.shape[0],self.shape[1]))
            int generation,index_sequence
        if start_equi==0:
            sequence = np.random.randint(0,high=self.Number_state_spin
                                 ,size = (self.Number_of_Node)
                                 ,dtype = np.int8)
        else:
            sequence = self.msa_no_phylo(1)[0]
        msa[0] = sequence
        for generation in range(1,n_generations+1):
            msa[int(2**(generation-1)):int(2**generation),:] = msa[0:int(2**(generation-1)),:]
            for index_sequence in range(int(2**generation)):
                self.mcmc(n_mutations_generation, msa[index_sequence], tottotH)
        return np.asarray(msa)
    
    ### Use of Fast tree ######
    def msa_tree_phylo(self, clade_root, int start_equi, char[:,::1] msa_real = None, double neff = 1.0, double theta = 0.2, double deltaMeff = 10):
        """
        clade_root, start_equi, Meff, neff = None, , theta = 0.2
        """   
        cdef:
            char[::1] first_sequence
            double Meff_tree, Meff, neff_o
            #char[:,::1] msa = np.random.randint(0,high=self.Number_state_spin, 
            #                                    size = (len(clade_root.get_terminals()),self.Number_of_Node)
            #                                    ,dtype = np.int8)
            char[:,::1] msa = np.zeros( (len(clade_root.get_terminals()),self.Number_of_Node), dtype=np.int8)
        if start_equi==0:
            print("root is a random sequence")
            first_sequence = np.random.randint(0,high=self.Number_state_spin
                                 ,size = (self.Number_of_Node)
                                 ,dtype = np.int8)
        else:
            print("root is a equilibrium sequence")
            first_sequence = self.msa_no_phylo(1)[0]
        if msa_real is not None :
            Meff = compute_Meff(msa_real, theta)
            print("Look for the best neff : Meff = %s"%Meff)
            neff = 1
            #msa = []
            msa = self._msa_tree_phylo_recur(clade_root, first_sequence, msa, neff)
            print("msa tree done")
            Meff_tree = compute_Meff(msa, theta)
            print("neff = %s : Meff_tree = %s"%(neff,Meff_tree))
            print("Look for the interval")
            if Meff_tree < Meff:
                while Meff_tree < Meff:
                    neff_o = neff
                    neff = 2*neff
                    msa = self._msa_tree_phylo_recur(clade_root, first_sequence, msa, neff)
                    Meff_tree = compute_Meff(msa, theta)
                    print("neff = %s : Meff_tree = %s"%(neff,Meff_tree))
                print("Dichotomie")
                while abs(Meff_tree - Meff)>deltaMeff:                   
                    neff_n = (neff_o + neff)/2
                    msa = self._msa_tree_phylo_recur(clade_root, first_sequence, msa, neff_n)
                    Meff_tree = compute_Meff(msa, theta)
                    print("neff = %s : Meff_tree = %s"%(neff_n,Meff_tree))
                    if Meff_tree>Meff:
                        neff = neff_n
                    else:
                        neff_o = neff
            
                return np.asarray(msa)
        else:
            return np.asarray(self._msa_tree_phylo_recur(clade_root, first_sequence, msa, neff))

    
    cdef char[:,::1] _msa_tree_phylo_recur(self, clade_root, char[::1] previous_sequence, char[:,::1] msa, double neff):
        cdef:
            char[::1] new_sequence = np.zeros((previous_sequence.shape[0]),dtype=np.int8)
            double[:,::1] tottotH = np.zeros((self.Field.shape[0],self.Field.shape[1]))
            int n_mutations
        b = clade_root.clades
        if len(b)>0:
            for clade in b:
                #Mutation on previous_sequences
                new_sequence[:] = previous_sequence
                n_mutations = int(clade.branch_length*new_sequence.shape[0]*neff)
                self.mcmc(n_mutations, new_sequence, tottotH)
                self._msa_tree_phylo_recur(clade, new_sequence, msa, neff)
        else:
            n_mutations = int(clade_root.branch_length*previous_sequence.shape[0]*neff)
            self.mcmc(n_mutations, previous_sequence, tottotH)
            msa[int(clade_root.name),:] = previous_sequence
        return msa
    

    #### standard function ########
    cdef void mcmc(self, int Number_of_Mutation, char[::1] sequence, double[:,::1] tottotH ) nogil:  
        cdef:
            int new_state, old_state, selected_node, c_mutation = 0
            double prob,prob_old,prob_new
        while c_mutation<Number_of_Mutation:   
            selected_node = randint(0,self.Number_of_Node)
            new_state = randint(0,self.Number_state_spin-1)
            old_state = sequence[selected_node]
            if new_state >= sequence[selected_node]:
                new_state += 1    
            prob_old = self._prob_sequence_cython(sequence, tottotH)
            if prob_old == 0:
                prob = 1.0
            else:
                sequence[selected_node] = new_state
                prob_new = self._prob_sequence_cython(sequence, tottotH)
                prob = prob_new/prob_old
            #prob = self.ratio_prob_sequence_cython(sequence, sequence_p)
            if prob>=1.0 or drand48()<prob:
                sequence[selected_node] = new_state
                c_mutation += 1
            else:
                sequence[selected_node] = old_state
                
    cdef inline double _prob_sequence_cython(self, char[::1] sequence, double[:,::1] tottotH ) nogil:
        cdef:
            int site, site1, a
            int q = self.p0.shape[0], N = self.Field.shape[0]
            #double[:,::1] tottotH = self.Field.copy()
            double z, max_totH, prob = self.p0[sequence[self.idxperm[0]]]  
        tottotH[:] = self.Field
        if prob == 0.0:
            pass
        else:
            for site in range(N):#Loop on index-1 already
                #totH[:] = self.Field[site] #copy memoryview
                for site1 in range(site+1): 
                    for a in range(q):
                        tottotH[site,a] += self.Coupling[site,a,int(sequence[self.idxperm[site1]]),site1]
                max_totH = max_list(tottotH[site])
                z = sum_exp(tottotH[site], max_totH)
                prob *= exp(tottotH[site,sequence[self.idxperm[site+1]]] - max_totH)/z
        return prob
    
    def prob_sequence_cython(self, char[::1] sequence):
        cdef:
            int site, site1, a
            int q = self.p0.shape[0], N = self.Field.shape[0]
            double[:,::1] tottotH = self.Field.copy()
            double z, prob = self.p0[sequence[self.idxperm[0]]] ##Check if it is the good way ? 
        tottotH[:] = self.Field
        if prob == 0.0:
            pass
        else:
            for site in range(N):#Loop on index-1 already
                #totH[:] = self.Field[site] #copy memoryview
                for site1 in range(site+1): 
                    for a in range(q):
                        tottotH[site,a] += self.Coupling[site,a,int(sequence[self.idxperm[site1]]),site1]
                prob *= softmax(tottotH[site])[sequence[self.idxperm[site1+1]]]
        return prob

cdef inline double sum_exp(double[::1] x, double max_x = 0.0) nogil:
    cdef : 
        int i
        double sumx = 0.0 
    for i in range(x.shape[0]):
        sumx += exp(x[i] - max_x)
    return sumx

cdef inline double[::1] softmax(double[::1] x) nogil:
    cdef : 
        int i
        double sumx = 0.0, max_x = max_list(x)
    for i in range(x.shape[0]):
        x[i] = exp(x[i] - max_x)
        sumx += x[i]
    for i in range(x.shape[0]):
        x[i]/=sumx
    return x

cdef inline double max_list(double[::1] x) nogil:
    cdef:
        double max_x = x[0]
        int i
    for i in range(1,x.shape[0]):
        if x[i]>max_x:
            max_x = x[i]
    return max_x

cdef inline int randint(int lower, int upper) nogil:
    """lower included, upper excluded"""
    return random() % (upper - lower ) + lower

cdef inline int rand_choice(double[::1] p) nogil:
    cdef:
        int i
        float randn
        double c = 0.0
    randn = drand48()
    for i in range(p.shape[0]):
        c += p[i]
        if randn<=c:
            return i
        
cpdef double compute_Meff(char[:,::1] msa, double theta):
    cdef:
        double[::1] sum_dist = np.zeros((msa.shape[0]))
        double Meff = 0.0
        int i,j
    if theta>0:
        with nogil:
            for i in prange(msa.shape[0]-1):
                for j in range(i+1,msa.shape[0]):
                    if hamming_dist(msa[i],msa[j])<theta:
                        sum_dist[i] += 1
                        sum_dist[j] += 1
    for i in range(msa.shape[0]):
        Meff += 1/(sum_dist[i]+1) #dont take into account the diagonal
    return Meff

cdef inline double hamming_dist(char[::1] a, char[::1] b) nogil:
    cdef:
        int i
        double dist = 0.0
    for i in range(a.shape[0]):
        dist+= (a[i]!=b[i])*1.0
    return dist/a.shape[0]