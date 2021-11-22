from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import uuid
import numpy as np

def write_msa_to_fasta(msa, file_name : str, alphabet : str):
    msa = np.array(msa)
    size_msa = msa.shape[0]
    print(file_name)
    with open(file_name, 'w') as f_out:
        
        for i in range(size_msa):
            sequence_str = ""
            sequence_num = msa[i]
            
            for amino in sequence_num:
                sequence_str += alphabet[amino]
                name = str(uuid.uuid1())
                
            record = SeqRecord(
                Seq(sequence_str),
                id=name,
                name="dss",
                description="dsad")
            SeqIO.write(record, f_out, "fasta")


