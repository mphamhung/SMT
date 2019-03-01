from align_ibm1 import *


e,f = read_hansard('/u/cs401/A2_SMT/data/Hansard/Testing/', 9000)

print(len(e))

for i in range(len(e)):
    print(e[i],f[i])
