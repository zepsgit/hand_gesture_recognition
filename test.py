from time import time


import time
k=50000000
s=time.time()
while k:
    k-=1
end=time.time()
print(end-s)