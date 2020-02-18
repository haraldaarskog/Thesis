import numpy as np

mat=np.matrix([
[0.0, 0.6, 0.4],
[0.7, 0.1, 0.2],
[0.3, 0.3, 0.4]
])

path=[0,2,1,2,2]
def aa(act):
    sum=1
    prevEl=0
    counter=0
    for element in path:
        if element==0:
            prevEl=element
            continue
        if counter==act:
            break
        sum*=mat[prevEl,element]
        prevEl=element
        counter+=1
    return sum

arr=[]
for i in range(1,len(path)):
    arr.append(aa(i))

print(50*arr)
