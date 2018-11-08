'''
Takes in files of lists and generates output with union and intersect

python union_intersect.py SAVE_NAME file1 file2 file3 ... filen
'''


import pandas as pd
import numpy as np
import sys, os

if len(sys.argv) <= 1:
    print(__doc__)
    exit()

save_name = sys.argv[1]

files = []
for i in range (2,len(sys.argv)):
    files.append(sys.argv[i])

d = []
count = 0
for fx in files:
  with open(fx) as f:
    file_list = f.read().splitlines()
  d.append(file_list)
  count += 1


#  Get union
d_union = set().union(*d)
print('Length of union: %i' % len(d_union))

#  Get intersection (present in all lists)
d_int = set.intersection(*map(set, d))
print('Length of intersect: %i' % len(d_int))

#  Get int3 (present in 3 lists...)
d_all = [item for sublist in d for item in sublist]
d_i2 = {x:d_all.count(x) for x in d_all}


save_union = save_name + '_UN'
save_int = save_name + '_IN'
save_i2 = save_name + '_I2'

with open(save_union, 'w') as out:
	for u in d_union:
		out.write('%s\n' % u)
out.close()

'''
with open(save_int, 'w') as out:
	for i in d_int:
		out.write('%s\n' % i)
out.close()
'''
c = 0
with open(save_i2, 'w') as out:
  for i in d_i2:
    if d_i2[i] >= 2:
      c += 1
      out.write('%s\n' % i)
out.close()
print('Length of intersect (>1): %i' % c)

print('Done!')