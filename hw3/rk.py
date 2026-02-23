import numpy as np
from scipy.optimize import fsolve

def plot_RAS(
    A,
    b,
    window_size= 3,
    num_points=1000,
    filled=False,
    ax=None,
    color='k',
    label=None,
    decorations=False
):
    '''
    Plots the region of absolute stability for an RK method with the given
    Butcher's array `A` and `b`.
    '''

    #### actually computing the contour ####

    # we can get the number of stages from the size of A
    s= A.shape[0]
    ones= np.ones(s)
    I= np.eye(s)

    # mesh grid of z values
    x= np.linspace(-window_size, window_size, num_points)
    y= np.linspace(-window_size, window_size, num_points)
    X,Y= np.meshgrid(x,y)

    # here z = h * lambda
    Z= X + (1j * Y)

    # we want to compute R(z) = 1 + z b^T (I - zA)^{-1} 1
    # we flatten Z and broadcast everything
    Z_flat= Z.ravel()
    LHS= I - Z_flat[:, None, None] * A
    # this inverts LHS
    sol= np.linalg.solve(LHS, ones)
    # we need to sol @ b to preserve the shapes
    R_flat= 1 + Z_flat * (sol @ b)
    R= R_flat.reshape(Z.shape)
    magnitude= np.abs(R)

    #### plotting ####
    assert ax != None, 'pass axes idiot'

    if filled: # this is easier to see what is inside vs outside, for testing
        ax.contourf(X,Y,magnitude, levels=[0, 1], colors='r')
    else:
        ax.contour(X,Y,magnitude, levels=[1], colors=color)

        # doing this nonsense to have something show up in the legend
        # too lazy to figure out how to do it the right way
        ax.plot([], [], color=color, label=label)
    
    # add some decorations here
    if decorations:
        ax.vlines([0], -window_size, window_size, colors='k', linestyles='dashed')
        ax.hlines([0], -window_size, window_size, colors='k', linestyles='dashed')
        ax.set(
            xlabel='$\\mathrm{Re}(h \\lambda)$',
            ylabel='$\\mathrm{Im}(h \\lambda)$'
        )
        ax.set_aspect('equal')

# A and b matrices for DIRK2 and DIRKo3

g_DIRK2= 1 - 1 / np.sqrt(2)
A_DIRK2= np.array([
    [g_DIRK2, 0],
    [1-g_DIRK2, g_DIRK2]
])
b_DIRK2= np.array([1-g_DIRK2, g_DIRK2])

g_DIRKo3= 1/2 + np.sqrt(3)/6
A_DIRKo3= np.array([
    [g_DIRKo3, 0],
    [1-2*g_DIRKo3, g_DIRKo3]
])
b_DIRKo3= np.array([1/2, 1/2])


def dirk(f, t_span, y0, method, h, A=None, b=None, dfdy=None):
    '''
    Diagonally Implicit Runge-Kutta with given A and b matrices.
    y0 always needs to be a 1d numpy array or it blows up.
    '''
    # pick the method pased on the `method` parameter
    if method == 'DIRK2':
        A= A_DIRK2
        b= b_DIRK2
    elif method == 'DIRKo3':
        A= A_DIRKo3
        b= b_DIRKo3
    elif method == 'custom':
        assert A != None and b != None, 'No A and b matrices provided'
    else:
        raise ValueError('Invalid method name')

    num_stages,cols= A.shape

    # validate the input a little bit, for fun
    if (num_stages != cols) or (b.shape != (num_stages,)): 
        raise ValueError(f'Invalid Dimensions. {A.shape=} while {b.shape=}')
    y0= np.asarray(y0)
    if len(y0.shape) > 1:
        raise ValueError(f'Invalid Dimensions. {y0.shape=}, must be a 1D array')

    # the c matrix can be derived from A
    c= np.sum(A, axis=1)
    
    # this is the ending time we're going to 
    t= np.arange(t_span[0], t_span[1], h)
    n_max= len(t)

    assert n_max == len(t), f'{n_max=} but {len(t)=}'


    # initialize u array to hold our values
    # I think this botched way of writing it will avoid having it be 2d in the case where y is scalar
    u= np.zeros([n_max] + list(y0.shape))
    print(f'{u.shape=}')
    u[0]= y0

    # compute the values for each time
    for n in range(1, n_max):
        if n % 100000 == 0:
            print(f'{n=}')
        # first, compute the k vector
        k= np.zeros(num_stages)
        for j in range(num_stages):
            # our initial guess for the solver will simply be f(t,u) (since kj are attempting to approximate this)
            x0= f(t[n-1], u[n-1])
            # note that A[j, :j] DOESN'T include A[j,j], :j is not inclusive in Python (I always forget)
            F= lambda kj: (kj - f(t[n-1] + c[j]*h, u[n-1] + h * (A[j, :j] @ k[:j] + A[j,j] * kj)))
            # if a derivative was provided, use it in our solution
            if dfdy != None:
                Fprime= lambda kj: (1 - dfdy(t[n-1] + c[j]*h, u[n-1] + h*(A[j, :j] @ k[:j] + A[j,j] * kj)) * h * A[j,j])
                k[j]= fsolve(
                    func=F,
                    x0=x0,
                    fprime=Fprime,
                    xtol=1e-4,
                    maxfev=20
                )
            # if not, use this torturously slow general purpose method
            else:
                k[j]= fsolve(F, x0)

        # then, perform the update
        u[n]= u[n-1] + h * b.T @ k

    return t, u

def newton(f, fprime, x0, max_iter=20, atol=1e-6):
    '''
    Approximate the solution to f(x)=0 using Newton's method
    '''
    # check if x0 is a scalar or vector
    x0= np.asarray(x0)
    j= 0 
    x=x0
    while j < max_iter and f(x) > atol:
        # if j % 100 == 0:
        #     print(f'Newton Iteration {j}')
        if len(x0) == 0:
            x += - f(x) / fprime(x)
        else:
            x += -np.linalg.solve(fprime(x), f(x))
        j += 1
    return x

def check_order(A, b):
    '''
    Check the order of a Runge-Kutta method up to order 3,
    using the conditions.
    '''

    (m,n)= A.shape
    assert m == n and b.shape == (n, ), 'improper dimenensions'

    # 1st order check
    if not np.isclose(np.einsum('l->', b), 1):
        return 0

    # 2nd order check
    if not np.isclose(2 * np.einsum('l,lq->', b, A), 1):
        return 1

    # 3rd order check
    if not (np.isclose(3 * np.einsum('l, lq, lr->', b, A, A), 1) \
        and np.isclose(6 * np.einsum('l, lq, qr->', b, A, A), 1)):
        return 2
    
    return 3
