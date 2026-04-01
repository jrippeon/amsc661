import numpy as np
import scipy

def __M(verts):
    '''
    `verts` is a 3x2 array containing the 3 corners of a triangle.

    This function calculates the M_{ij} matrix containing the contribution of this triangle
    to the A_{ij} of each of the pairs of corners.
    G= Aux^{-1} @ rhs
    M= |T| * G @ G.T = 1/2 * det(Aux) * G @ G.T
    '''

    Aux= np.ones((3,3))
    Aux[1:3, :]= verts.T
    rhs= np.zeros((3,2))
    rhs[1,0]= 1
    rhs[2,1]= 1
    G= np.zeros((3,2))
    G[:,0]= np.linalg.solve(Aux, rhs[:,0])
    G[:,1]= np.linalg.solve(Aux, rhs[:,1])
    M= 0.5 * np.linalg.det(Aux) * G @ G.T
    return M

    

def laplace(pts, tri, dirichlet_bdy_segments):
    '''
    Solve Δu = 0 on a specified mesh.

    `dirichlet_bdy_segments` is a list of tuples of (list of point indices, value)
    '''
    Npts= pts.shape[0]
    Ntri= tri.shape[0]

    A= np.zeros((Npts, Npts), dtype=float)
    b= np.zeros(Npts, dtype=float)
    u= np.zeros(Npts, dtype=float)

    # we will update this with the points on the boundary as they are added from each segment
    bdy_indices_list= []

    for t in dirichlet_bdy_segments:
        seg_pts, seg_val= t
        bdy_indices_list.append(seg_pts)
        # update u with known values
        u[seg_pts]= seg_val
    # this glues them all together
    dirichlet_bdy_indices= np.concatenate(bdy_indices_list, axis=None)
    
    # get the indices of the free points by taking a set difference with the boundary points
    free_indices= np.setdiff1d(np.arange(0, Npts, 1, dtype=int), dirichlet_bdy_indices, assume_unique=True)

    # populate the A matrix triangle by triangle
    for j in range(Ntri):
        verts= pts[tri[j,:], :] # vertices of this triangle
        idxs= tri[j,:]
        # this is how we can index into a submatrix
        A[np.ix_(idxs, idxs)] += __M(verts)

    # populate the load vector with known values
    b -= A @ u

    # solve for the free values in u
    # A_sparse= scipy.sparse.csr_matrix(A[np.ix_(free_indices, free_indices)])
    # u[free_indices]= scipy.sparse.spsolve(A_sparse, b[free_indices])
    u[free_indices]= np.linalg.solve(A[np.ix_(free_indices, free_indices)], b[free_indices])

    return u
        

def laplace_advanced(pts, tri, a, dirichlet_bdy_segments):
    '''
    Solve -∇ * a ∇u = 0 on a specified mesh.

    `dirichlet_bdy_segments` is a list of tuples of (list of point indices, value)
    `a` is a function from points to a real value

    Any boundary segments without dirichlet conditions specified get homogeneous Neumann conditions.
    '''
    Npts= pts.shape[0]
    Ntri= tri.shape[0]

    A= np.zeros((Npts, Npts), dtype=float)
    b= np.zeros(Npts, dtype=float)
    u= np.zeros(Npts, dtype=float)

    # we will update this with the points on the boundary as they are added from each segment
    bdy_indices_list= []

    for t in dirichlet_bdy_segments:
        seg_pts, seg_val= t
        bdy_indices_list.append(seg_pts)
        # update u with known values
        u[seg_pts]= seg_val
    # this glues them all together
    dirichlet_bdy_indices= np.concatenate(bdy_indices_list, axis=None)
    
    # get the indices of the free points by taking a set difference with the boundary points
    free_indices= np.setdiff1d(np.arange(0, Npts, 1, dtype=int), dirichlet_bdy_indices, assume_unique=True)

    # populate the A matrix triangle by triangle
    for j in range(Ntri):
        verts= pts[tri[j,:], :] # vertices of this triangle
        midpoint = np.sum(verts, axis=0) / 3
        idxs= tri[j,:]
        # this is how we can index into a submatrix
        # we simply need to multiply the result from __M by the value of a at the midpoint
        A[np.ix_(idxs, idxs)] += a(midpoint) * __M(verts)

    # populate the load vector with known values
    b -= A @ u

    # solve for the free values in u
    # A_sparse= scipy.sparse.csr_matrix(A[np.ix_(free_indices, free_indices)])
    # u[free_indices]= scipy.sparse.spsolve(A_sparse, b[free_indices])
    u[free_indices]= np.linalg.solve(A[np.ix_(free_indices, free_indices)], b[free_indices])

    return u
        
def grad(pts, tri, u, faces=False):
    '''
    Returns the gradient of u at each vertex.

    If `faces == True` then instead of returning the gradient at each vertex,
    a list of the gradient inside each triangle is returned.
    '''

    Ntris= tri.shape[0]
    Npts= pts.shape[0]

    # first, compute the gradient in the center of each triangle
    A= np.array([
        [0, 1, 0],
        [0, 0, 1]
    ])
    N_adjacent_triangles= np.zeros(Npts)
    du= np.zeros(pts.shape)
    du_faces= np.zeros((Ntris, 2))
    for j in range(Ntris):
        indices= tri[j,:]
        verts= pts[indices, :]
        us= u[indices]
        # this is a formula from the homework
        B= np.ones((3,3))
        B[:, 1:3]= verts
        du_triangle_center= A @ np.linalg.solve(B, us)
        du_faces[j, :]= du_triangle_center

        # add this du to each adjacent triangle
        du[indices,:] += du_triangle_center

        # count the additional triangle
        N_adjacent_triangles[indices] += 1
    
    # divide by the number of adjacent triangles
    du /= N_adjacent_triangles[:, None] # turn it upright for broadcasting

    return du_faces if faces else du


        