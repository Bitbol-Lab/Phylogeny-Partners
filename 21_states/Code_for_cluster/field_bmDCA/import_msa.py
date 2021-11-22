import numpy as np

def import_msa_bmDCA(path_file):
    L = 176
    J2 = np.zeros((L, L, 21, 21), dtype=np.float64)
    h = np.zeros((L, 21), dtype=np.float64)
    with open(path_file, "r") as f:
        for line in f:
            l = line.rstrip("\n").split(" ")
            val = float(l[-1]) 
            if l[0] == "J":
                J2[int(l[1]), int(l[2]), int(l[3]), int(l[4])] = val
            elif l[0] == "h":
                h[int(l[1]), int(l[2])] = val
    # Symmetrize J2
    for i in range(J2.shape[0]):
        for j in range(i):
            J2[i, j, ...] = J2[j, i, ...].T
    return h,J2
