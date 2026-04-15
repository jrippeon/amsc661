import numpy as np
from math import factorial
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


### PROCEDURE CHANGE ###
# I'm going to store our A and B matrices in a "local" format, instead of flattening
# the matrices to (Npts, Npts) I will keep them as (Ntri, 3, 3)
# This will make things a little more conceptually clear I think

def triangle_matrix(pts, tri):
    '''
    Returns (Ntri, 3, 3) array of the matrix

        [ 1   1   1]
    T = [x1  x2  x2]
        [y1  y2  y2]

    This is useful where triangles are concerned
    '''
    Npts= pts.shape[0]
    Ntri= tri.shape[0]
    vertices= pts[tri]
    # G = [ 1   1   1]-1 [0 0]
    #     [x1  x2  x2]   [1 0]
    #     [y1  y2  y2]   [0 1]
    T= np.ones((Ntri,3,3))
    T[:, 1:, :]= np.linalg.matrix_transpose(vertices)

    return T

def A_local(pts, tri):
    '''
    Given a triangulation, construct the A_local array of shape (Ntri, 3, 3)
    A_nij = ∫_{T_n} ∇η_i * ∇η_j dx
    '''
    # G = [ 1   1   1]-1 [0 0]
    #     [x1  x2  x2]   [1 0]
    #     [y1  y2  y2]   [0 1]

    T= triangle_matrix(pts, tri)

    T_inv= np.linalg.inv(T)
    G= T_inv[:, :, 1:]
    dets= np.abs(np.linalg.det(T))
    # einsum is G @ G.T
    A= 0.5 * dets[:, None, None] * np.einsum('nik,njk->nij', G, G)

    return A

def B_local(pts, tri):
    '''
    Given a triangulation, construct the B_local array of shape (Ntri, 3, 3),
    B_nij = ∫_{T_n}  η_i * η_j dx
    '''
    Npts= pts.shape[0]
    Ntri= tri.shape[0]

    # (Ntri, 3, 2) array of the coordinates of each point
    vertices= pts[tri]
    
    # B =   1   [x2-x1  x3-x1]   [2 1 1]
    #      -- * [y2-y1  y3-y1] * [1 2 1]
    #      24                    [1 1 2]
    #                 Aux           C

    # None adds a new axis so that it broadcasts correctly across the rows
    Aux_T= vertices[:, 1:] - vertices[:, 0, None, :]
    Aux= np.linalg.matrix_transpose(Aux_T)

    C= np.array([ 
        [2,1,1],
        [1,2,1],
        [1,1,2] 
    ])

    dets= np.abs(np.linalg.det(Aux))

    B= 1/24 * dets[:, None, None] * C[None, :, :]

    return B

def C_local(pts, tri):
    '''
    Given a triangulation, construct the C_local array of shape (Ntri, 3, 3, 3, 3),
    C_nijkl = ∫_{T_n}  η_i * η_j * η_k * η_l dx

    Arises from a formula
    ∫_T  η_1^a * η_2^b * η_3^c dx = 2|T| /(a+b+c+2)! a!b!c!
    '''
    Npts= pts.shape[0]
    Ntri= tri.shape[0]

    # (Ntri, 3, 2) array of the coordinates of each point
    vertices= pts[tri]

    T= triangle_matrix(pts, tri)
    dets= np.abs(np.linalg.det(T))

    # construct the (3,3,3,3) tensor of various constants 
    coeffs= np.zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    indices=[i,j,k,l]
                    a= indices.count(0)
                    b= indices.count(1)
                    c= indices.count(2)
                    coeffs[i,j,k,l]= factorial(a) * factorial(b) * factorial(c) / 720

    C= dets[:, None, None, None, None] * coeffs[None, :, :, :, :]

    return C

def b_vector(pts, tri, u, eps, A, B, C):
    '''
    'b' vector of M w = b version of DJ(u_n, η_i; w) = J(u_n, η_i)

    I.e. b_i = ε∫ ∇u * ∇η_i dx - ∫ (u-u^3)η_i dx
             = ε Σ u_j A_ij - Σ u_j Bij + Σ u_j u_k u_l C_ijkl
    '''
    Npts= pts.shape[0]
    Ntri= tri.shape[0]
    # get the u value at each vertex, shape (Ntri, 3) 
    u_local= u[tri]
    Au= np.einsum('nij,nj->ni', A, u_local)
    Bu= np.einsum('nij,nj->ni', B, u_local)
    Cuuu= np.einsum('nijkl,nj,nk,nl-> ni', C, u_local, u_local, u_local)

    b_local= eps * Au - Bu + Cuuu

    # now assemble into one b vector
    b= np.zeros(Npts)

    # flatten tri and b_local from (Ntri, 3) to (Ntri*3,)
    indices= tri.flatten() 
    b_local_flat= b_local.flatten()
    # add b_local entries to b at the indices from tri
    np.add.at(b, indices, b_local_flat)

    return b


def M_matrix(pts, tri, u, eps, A, B, C):
    '''
    'M' matrix of M w = b version of DJ(u_n, η_i; w) = J(u_n, η_i)

    I.e. Mw = ε∫ ∇η_i * ∇w dx - ∫ (1-3u^2) η_iw dx
            = Σ w_j (ε A_ij - B_ij + 3 Σ u_k u_l C_ijkl)
    so   M_ij = ε A_ij - B_ij + 3 Σ u_k u_l C_ijkl
    '''
    Npts= pts.shape[0]
    Ntri= tri.shape[0]

    u_local= u[tri]
    Cuu= np.einsum('nijkl,nk,nl->nij', C, u_local, u_local)

    M_local= eps * A - B + 3 * Cuu

    # flatten the array (AI told me how to do this efficiently lol)
    row_indices= np.zeros((Ntri, 3, 3), dtype=int)
    col_indices= np.zeros((Ntri, 3, 3), dtype=int)

    # basically we want to make a meshgrid so we know which row and column of the
    # full M matrix each row and column of (each layer of) M_local belongs to 
    for i in range(3):
        for j in range(3):
            row_indices[:, i, j]= tri[:,i]
            col_indices[:, i, j]= tri[:,j]

    # next we flatten everything and put it in a coo_matrix
    # this has the benefit that any repeated indices just get summed over,
    # which is exactly the behavior we wanted
    M_local_flat= M_local.flatten()
    row_indices_flat= row_indices.flatten()
    col_indices_flat= col_indices.flatten()
    M_coo= coo_matrix((M_local_flat, (row_indices_flat, col_indices_flat)), shape=(Npts, Npts))

    # this format can be put into scipy.sparse.linalge.spsolve()
    M= M_coo.tocsr()

    return M

def newton(J, DJ, u_init, interior, max_iter=50, atol=1e-10):
    '''
    Newtons method, to solve J(u) = 0.

    We convert DJ(u_n, η_i; w) = J(u_n, η_i) into M w = b then solve.
    DJ(u) should be the matrix M, and J(u) the vector b
    '''
    u= np.copy(u_init)

    # only update the interior nodes, we don't want to change the boundary
    for i in range(max_iter):
        M= DJ(u)
        M_int= M[np.ix_(interior, interior)] 
        b= J(u)
        b_int= b[interior]
        w_int= spsolve(M_int, b_int)
        u[interior]= u[interior] - w_int

        if np.linalg.norm(w_int) < atol:
            break
    
    return u

def ginzburg_landau(pts, tri, boundary, uD=None, u_init=None, eps=1e-2):
    '''
    Solve the (Nonlinear) Ginzburg-Landau equation on the specified mesh.

    uD is the Dirichlet boundary condition, if omitted then assumed homogeneous.

    u_init is the choice of initial condition for Newton iteration, if unspecified
    then an arbitrary choice will be made.

    J(u,v) = ε∫ ∇u * ∇v dx - ∫ (u-u^3)v dx
    DJ(u,v;w)= ε∫ ∇v * ∇w dx - ∫ (1-3u^2)vw dx

    We will solve J(u, η_i) = 0 numerically, using Newton's method.
    '''
    Npts= pts.shape[0]
    Ntri= tri.shape[0]

    # we only need to compute these once
    A= A_local(pts, tri)
    B= B_local(pts, tri)
    C= C_local(pts, tri)

    # get the indices of the interior
    interior= np.setdiff1d(np.arange(Npts), boundary)

    # create J and DJ functions to pass to Newtons method
    def DJ(u):
        M= M_matrix(pts, tri, u, eps, A, B, C)
        return M
    def J(u):
        b= b_vector(pts, tri, u, eps, A, B, C)
        return b

    # if we didn't get boundary conditions, assume they're homogeneous
    if uD is None:
        uD= np.zeros(len(boundary))

    # if there was not a provided initial vector, just initialize with noise lol
    if u_init is None:
        u_init= np.random.random(Npts)
        u_init[boundary]= uD

    u= newton(J, DJ, u_init, interior)

    return u
