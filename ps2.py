from qutip import *
from numpy import *
N = 6
phi1 = basis(N, 1)
phi2 = basis(N, 1)

W1 = 1000000000
W2 = 1000000000
X12 = 1000000
T = pi/X12
def getH(w1, w2, x12):
    p1 = w1*tensor(create(N)*destroy(N), qeye(2))
    p2 = w2*tensor(qeye(N), basis(2,1).proj())
    p3 = x12*tensor(create(N)*destroy(N), basis(2,1).proj())
    return p1+p2-p3

beta = 2.0
gg = Qobj([[1,1],[1,1]])
print(gg)
initial_state = tensor(basis(N,0), basis(2,0))
print(initial_state)
O1 = tensor(qeye(N), Qobj([[1,1],[1,1]])/sqrt(2))
print(O1*initial_state)
D1 = tensor(displace(N, beta), qeye(2))

s1 = D1*(O1*initial_state)

H = getH(W1,W2,X12)
print(s1)
result = mesolve(H, s1, linspace(0, T, 1000), [],[])
print(result.states[-1])
s2 = result.states[-1]
O2 = tensor(qeye(N), qeye(2)) + tensor(basis(N,0).proj(), sigmax() - qeye(2))
D2 = tensor(displace(N, -beta), qeye(2))
s3 = D2*O2*D1*s2
print(s3)
print(s3.norm())

gg2 = coherent(N, beta) + coherent(N, -beta)
print(gg2)
expected = tensor(coherent(N, beta) + coherent(N, -beta), basis(2,0))

print(expected)
print((expected - s3).norm())