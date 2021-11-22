from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import random
import sys

def write_new_tree_for_ccmgen(original_fasta_file : str, file_name : str):
    
    l = list(SeqIO.parse(original_fasta_file, "fasta"))
    size_prot = len(l[0])
    
    sequence = ''.join([random.choice('ACDEFGHIKLMNPQRSTVWY-') for x in range(size_prot)]) 

    print("Random sequence : %s"%sequence)
    
    record = SeqRecord(
        Seq(sequence),
        id="ID_0.1",
        name="RandomSequences",
        description="Random Sequences for the root of a phylogeny tree",
    )

    SeqIO.write(record, file_name, "fasta")

if __name__ == '__main__':
    # Map command line arguments to function arguments.
    write_new_tree_for_ccmgen(*sys.argv[1:])
