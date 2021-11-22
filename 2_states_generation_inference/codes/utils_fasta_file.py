from Bio import SeqIO
import numpy as np

def dictionnary_species(file_fasta):
    """
    return d_species
    """
    d_species = {}
    for ind,record in enumerate(SeqIO.parse(file_fasta, "fasta")):
        name = record.id[record.id.find('|')+1:]
        name = name[:name.find('|')]
        if name not in d_species.keys():
             d_species[name] = [ind]
        else:
            d_species[name].append(ind)
    return d_species

def list_name_prot(file_fasta):
    """
    return list of name ordered as the fasta file
    """
    l_species = []
    for ind,record in enumerate(SeqIO.parse(file_fasta, "fasta")):
        name = record.id
        l_species.append(name)
    return l_species

def get_msa_fasta_file(file_fasta):
    all_amino = 'ACDEFGHIKLMNPQRSTVWY-'
    msa=[]
    for record in SeqIO.parse(file_fasta, "fasta"):
        l=[]
        for amino in record.seq:
            index = all_amino.find(amino)
            l.append(index)
        msa.append(l)
    msa=np.array(msa,dtype=np.int8)
    return msa