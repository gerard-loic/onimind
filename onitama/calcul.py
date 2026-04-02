import math

C = 1.5
Q = 0.25
P = 0.5
N = 2
Np = 2

PUCT = (C*P)+(math.sqrt(Np)/(1+N))

print(PUCT)