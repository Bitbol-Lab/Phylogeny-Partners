import uuid
from Bio import Phylo
import sys
import extract_msa as extr

## Modify the tree infered by FastTree2 to allow the use of this tree by CCMgen
## See the usage in generation_sequence_CCMgen.run

def modify_clades(clade_root, l_prot_name):
    b = clade_root.clades
    if len(b)>0:
        for clade in b:
            if clade.name is None:
                clade.name = str(uuid.uuid1())
            modify_clades(clade, l_prot_name)
    else:
        i = find_index_name(l_prot_name, clade_root.name)
        clade_root.name = str(i)

def write_new_tree_for_ccmgen(file_tree, file_new_tree, file_fasta):
    print("Tree used : ", file_tree)
    tree = Phylo.read(file_tree,"newick")
    l_prot_name = extr.list_name_prot(file_fasta)
    modify_clades(tree.clade, l_prot_name)
    tree.clade.name = "root"
    tree.clade.branch_length = 0
    print("New Tree to use : ", file_new_tree)
    tree = Phylo.write(tree, file_new_tree, "newick")

def find_index_name(l_name_prot, name_clade):
    for ind, name in enumerate(l_name_prot):
        if name_clade in name:
            return ind
    raise ValueError("Name of the leafes of the tree not found")

if __name__ == '__main__':
    # Map command line arguments to function arguments.
    write_new_tree_for_ccmgen(*sys.argv[1:])
