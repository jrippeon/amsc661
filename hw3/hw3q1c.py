from sympy import symbols, Matrix, eye, ones, simplify, latex, pprint, S, collect

gamma, z= symbols('γ z')

A= Matrix([
    [gamma, 0],
    [1 - 2 * gamma, gamma]
])
# the S(1) makes 1 a sympy integer so this ends up as 1/2 instead of 0.5
b= Matrix([
    [S(1) / 2],
    [S(1) / 2]
])
I= eye(2)
one= ones(2,1)

# the output of all these matrix operations is a 1x1 matrix, need to unwrap it
temp= z * b.T * ((I - z * A)**-1 * one)
R= 1 + temp[0,0]
R= simplify(R)
R= collect(R, z)

# pprint(R)
print(latex(R))
