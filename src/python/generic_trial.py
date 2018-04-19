
import multiprocessing
import math

l = [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
a = 'bnbn'

print(type(l))
print(type(a))

if isinstance(l, list):
    print('yes')
else:
    print('no')

print(math.floor(math.sqrt(2*multiprocessing.cpu_count())))
