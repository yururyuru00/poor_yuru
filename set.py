#!/usr/bin/env python
# coding: utf-8

# In[19]:


import re

file = 'C:/Users/yajima/Java/ResearchSNSClustering/multiview_data_20130124/rugby/rugby'
path_in = file + '.communities'
path_out = file + "$.communities"

ls = []
commu = []

with open(path_in, "r") as f:
    for line in f.readlines():
        commu.append(re.search(r"[a-z-]+", line).group())
        lines = re.sub('[a-z-]+: ', '', line).rstrip().split(',')
        l = set()
        for val in lines:
            l.add(val)
        ls.append(l)

for i in range(0, len(ls)):
    for j in range(i+1, len(ls)):
        print("(" + str(i) + "," + str(j) + ") ")
        if(len(ls[i] & ls[j]) > 0):
            print(ls[i]&ls[j])
            for id in (ls[i] & ls[j]):
                if(len(ls[i]) >= len(ls[j])):
                    ls[i].remove(id)
                else:
                    ls[j].remove(id)
        else:
            print("none")
           
"""
with open(path_out, "w") as w:
    for i in range(0, len(ls)):
        w.write(commu[i] + ": ")
        for id in ls[i]:
            w.write(str(id) + ",")
        w.write("\n")
"""


# In[ ]:




