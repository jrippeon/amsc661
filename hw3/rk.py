import numpy as np
from scipy.optimize import fsolve


# A and b matrices for DIRK2 and DIRKo3

g_DIRK2= 1 - 1 / np.sqrt(2)
A_DIRK2= np.array([
    [g_DIRK2, 0],
    [1-g_DIRK2, g_DIRK2]
])
b_DIRK2= np.array([1-g_DIRK2, g_DIRK2])
b_DIRK2_low_order= np.array([1, 0])

g_DIRKo3= 1/2 + np.sqrt(3)/6
A_DIRKo3= np.array([
    [g_DIRKo3, 0],
    [1-2*g_DIRKo3, g_DIRKo3]
])
b_DIRKo3= np.array([1/2, 1/2])


def dirk(f, t_span, y0, method, h, A=None, b=None, b_bad=None, dfdy=None, adaptive=False, atol= 1e-5, rtol=1e-5, order=None, progress=False, output_hist=False):
    '''
    Diagonally Implicit Runge-Kutta with given A and b matrices.
    '''

    ############################################################################

    # pick the method pased on the `method` parameter
    if method == 'DIRK2':
        A= A_DIRK2
        b= b_DIRK2
    elif method == 'DIRKo3':
        A= A_DIRKo3
        b= b_DIRKo3
    elif method == 'custom':
        assert A is not None and b is not None, 'No A and b matrices provided'
    else:
        raise ValueError('Invalid method name')

    # validate adaptive step size
    if adaptive:
        if method == 'DIRK2':
            b_bad= b_DIRK2_low_order
            order= 2
        elif method == 'custom' and (b_bad is None or order is None):
            raise ValueError('b_bad or order missing for custom method')
        else:
            raise ValueError(f'Adaptive step size not supported for {method}, implementation left as an exercise :^)')


    num_stages,cols= A.shape

    # validate the input a little bit, for fun
    if (num_stages != cols) or (b.shape != (num_stages,)): 
        raise ValueError(f'Invalid Dimensions. {A.shape=} while {b.shape=}')

    y0= np.array(y0, dtype=float)
    if y0.ndim > 1:
        raise ValueError(f'Invalid Dimensions. {y0.shape=}, must be a scalar or 1D array')

    ############################################################################

    # the c matrix can be derived from A
    c= np.sum(A, axis=1)
    

    # initialization depends on whether we are adaptive
    if adaptive:
        # random guess for the initial array length
        current_array_length= 1000
        t= np.zeros(current_array_length)
        t[0]= t_span[0]
        u= np.zeros((current_array_length,) + y0.shape)
        if output_hist:
            h_hist= np.zeros(current_array_length)
            h_hist[0]= h

    else:
        # Initialize the time array
        t= np.arange(t_span[0], t_span[1], h)
        n_max= len(t)
        # initialize u array to hold our values, we use this to handle the scalar and vector cases
        u= np.zeros((n_max,) + y0.shape)
    u[0]= y0

    # initialize an identity matrix, this might be a scalar depending
    I= 1.0 if (y0.ndim == 0) else np.eye(y0.shape[0])

    ############################################################################

    # I moved these outside the loop so I don't have to allocate them every time,
    # might do something, might not
    F= lambda kj, j, n, h, k: (kj - f(t[n-1] + c[j]*h, u[n-1] + h * (A[j, :j] @ k[:j] + A[j,j] * kj)))
    Fprime= lambda kj, j, n, h, k: (I - dfdy(t[n-1] + c[j]*h, u[n-1] + h*(A[j, :j] @ k[:j] + A[j,j] * kj)) * h * A[j,j])

    # this routine is common to both versions of the algorithm
    def calc_k(h):
        k= np.zeros((num_stages,) + y0.shape)
        k0= f(t[n-1], u[n-1])
        for j in range(num_stages):
            k[j]= newton(
                f=F,
                fprime=Fprime,
                x0=k0,
                args=(j, n, h, k)
            )
        return k


    if adaptive:
        n= 1
        while t[n-1] < t_span[1]:
            # first, resize the arrays if needed
            if n == current_array_length:
                current_array_length *= 2

                t_new= np.zeros(current_array_length)
                t_new[:n]= t[:n]
                t= t_new

                if output_hist:
                    h_new= np.zeros(current_array_length)
                    h_new[:n]= h_hist[:n]
                    h_hist= h_new

                u_new= np.zeros((current_array_length,) + y0.shape)
                u_new[:n]= u[:n]
                u= u_new
            
            # check the results with our current h
            k= calc_k(h)
            un_good= u[n-1] + h * b.T @ k
            un_bad= u[n-1] + h * b_bad.T @ k
            # infinity norm :D
            e= np.max(np.abs(un_good - un_bad))
            E= atol + rtol * np.max(np.abs(un_good))

            # if the acceptance criterion is satisfied, accept the current step
            if e < E:
                u[n]= un_good
                t[n]= t[n-1] + h
                if output_hist: h_hist[n]= h

                if progress and n % progress == 0:
                    print(f't={t[n]}')

                n += 1
            # either way, update to a new h
            # add a bit to e to avoid a divide by zero, and limit growth to a factor of 2 each time
            h *= (E / (e + 1e-15))**(1/order) * 0.9

        # note that we only return the parts of the array we wrote to
        if output_hist: return t[:n],u[:n],h_hist[:n]
        return t[:n],u[:n]

    else:
        # compute the values for each time
        for n in range(1, n_max):
            # if n % 100000 == 0:
            #     print(f'{n=}')
            # our initial guess for the solver will simply be f(t,u) (since kj are attempting to approximate this)
            k0= f(t[n-1], u[n-1])
            k= calc_k(h)
            u[n]= u[n-1] + h * b.T @ k

            if progress and n % progress == 0:
                print(f't={t[n]}')

        return t, u
            


def newton(f, fprime, x0, max_iter=20, atol=1e-12, args=()):
    '''
    Approximate the solution to f(x)=0 using Newton's method.
    '''
    # get a copy so we don't mutate x0
    x=np.array(x0, dtype=float)
    # check if x was a scalar so we return the same shape
    is_scalar= (x.ndim == 0)
    x= np.atleast_1d(x)

    for _ in range(max_iter):
        # unwrap x if necessary
        x_input= x[0] if is_scalar else x
        fx= np.atleast_1d(f(x_input, *args))
        # check current infinity norm
        if np.max(np.abs(fx)) < atol:
            break
        jx= np.atleast_2d(fprime(x_input, *args))
        x -= np.linalg.solve(jx, fx)
    # this is way too annoying
    # else:
    #     print(f'newton did not converge after {max_iter} iterations')

    return x[0] if is_scalar else x

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