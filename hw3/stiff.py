import numpy as np
import rk
import linear_multistep

def stiff(func, jacobian, t_span, y0, h, method):
    '''
    A number of good (and bad) solvers for stiff problems

    Currently available:
    DIRK2   - DIRK2
    DIRKo3  - Order 3 DIRK Method (bad)
    BDF2    - 2-step Backwards Differentiation Formula Method
    '''
    match method:
        case 'DIRK2':
            return rk.dirk(
                f=func, 
                t_span=t_span,
                y0=y0,
                method='DIRK2',
                h=h,
                dfdy=jacobian
            )
        case 'DIRKo3':
            return rk.dirk(
                f=func, 
                t_span=t_span,
                y0=y0,
                method='DIRKo3',
                h=h,
                dfdy=jacobian
            )
        case 'BDF2':
            # coefficients due to wikipedia et al
            a_bdf= np.array([-4/3, 1/3])
            b_bdf= np.array([0, 0])
            b_1_bdf= 6/11

            return linear_multistep.linear_multistep(
                func=func,
                dfdy=jacobian,
                t_span=t_span,
                y0=y0,
                h=h,
                a=a_bdf,
                b=b_bdf,
                b_1=b_1_bdf,
            )
        case _:
            raise ValueError(f"Method '{method}' not recognized.")
