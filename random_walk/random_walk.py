import numpy as np
import numpy.linalg as LA
import random
import sys

sys.setrecursionlimit(10000)

gfile = "./Linklist.txt" #リンクリストのファイル
n=1000  #ノード数
a=1  #探索要求ノード
b=100   #探索目的ノード 2,100,1000
n_a=10  #エージェント数
T=1000  #探索の試行回数
limit = 100000

A = np.zeros((n, n))

for line in open(gfile, 'r'):
    i, j,w= map(int, line.split())
    A[i-1, j-1] = w
    
link=[[] for i in range(len(A))]

for i in range(len(A)):
    for j in range(len(A)):
        if A[i][j]==1:
            link[i].append(j)

def Random(list,a,b):
    global number
    value=random.choice(list[a-1])
    #print('選択=',value)
    if value==b-1:
        number+=1
        #print(number)
        return number
    else:
        number+=1
        return Random(list,value,b)

def Random2(list, a, b):
    global number, limit
    for _ in range(limit):
        value=random.choice(list[a-1])
        #print('選択=',value)
        if value==b-1:
            number+=1
            #print(number)
            return number
        else:
            number+=1
            a = value
    return number
    

sum=0
for i in range(T):
    array=[]
    for j in range(n_a):
        number=0
        array.append(Random2(link,a,b))
    array.sort()
    sum+=array[0]
print('探索時間の平均値='+str(sum/T)+' b='+str(b)+'の場合')