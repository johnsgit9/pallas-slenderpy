"""Solver for a wake-oscillator force on a cable."""

import numpy as np
import scipy as sp
import random
from slenderpy import _cable_utils as cbu
from slenderpy import _progress_bar as spb
from slenderpy import cable
from slenderpy import force
from slenderpy import simtools
from slenderpy import turbwind


def solve(cb: cable.SCable,
          pm: simtools.Parameters,
          wo: force.WOP,
          md0: int = 1,
          dmd0: int = 0,
          y0: float = 0.,
          q0: float = 0.05,
          fast: bool = True) -> simtools.Results:
    """EOM solver for a cable and wake oscillator.

    Parameters
    ----------
    cb : slenderpy.cable.SCable
        A cable object.
    pm : slenderpy.simtools.Parameters
        Simulation parameters.
    wo : structib.force.WOP
        Wake oscillator parameters.
    md0 : int, optional
        Mode for initial cable shape. The default is 1.
    y0 : float, optional
        Amplitude in meters for initial cable shape and fluid variable. The
        default is 0.
    q0 : float, optional
        Amplitude (no unit) for initial fluid-cariable. The default is 0.05.
    fast : bool, optional
        Use faster version (work in progress). The default is True.

    Returns
    -------
    slenderpy.simtools.Results
        Simulation output with offsets (tangential, normal and binormal) and
        fluid variable for the positions and times specified in input parameters.

    """
    M = wo.rho * cb.d**2 * wo.cl0 / (16. * np.pi**2 * wo.st**2 * cb.m)
    if wo.tbw is None:
        w = 2. * np.pi * wo.u * wo.st / cb.d
    
        al = wo.al / w**2
        bt = wo.bt / w**2
        gm = wo.gm / w**2

    ns, s, ds, N, n = cbu.spacediscr(pm.ns)
    vt2, vl2 = cbu.vtvl(cb)
    C, A, I, J = cbu.matrix(ds, n)
    #tAd, uAd = cbu.adim(cb)
    # q has frequency 1rad/s
    tAd = 1. / w
    uAd = cb.L / tAd
    t, tf, dt, ht, ht2 = cbu.times(pm, tAd)
    z = np.zeros_like(s)
    _, ub, vn, vb = cbu.init_vars(cb, s, z, z, z, z, uAd, False)
    
    def init_sin(m, y0, q0, qconst=False, y0rand=False):
        if y0rand:
            ytmp = y0 * random.uniform(0.0, 1.0)
        else:
            ytmp = y0
        # initial displacement
        un = ytmp * np.sin(m * 2.0 * np.pi * s)
        # initial wake oscillator parameter
        if qconst:
            q = q0 * np.ones_like(s)
        else:
            q = q0 * np.sin(m * 2.0 * np.pi * s)
        return (un, q)
    
    if md0 > 0 and dmd0 > 0:
        # mutli-mode initialisation
        un = np.zeros((pm.ns,))
        q = np.zeros((pm.ns,))
        for m in range(md0, md0+dmd0+1):
            out = init_sin(m, y0, q0, qconst=False, y0rand=True)
            un += out[0]
            q += out[1]
    elif md0 > 0 and dmd0 == 0:
        # mono-mode initialisation
        out = init_sin(md0, y0, q0)
        un = out[0]
        q = out[1]
    else:
        # constant q=2.0
        un = np.zeros_like(s)
        q = 2.0 * np.ones_like(s)
        
    '''# initial displacement
    un = y0 * np.sin(md0 * 2.0 * np.pi * s)
    # initial wake oscillator parameter
    q = q0 * np.sin(md0 * 2.0 * np.pi * s)'''
    
    r = np.zeros_like(q)
    ut, ef = cbu.utef(un, ub, C, s, ds, vt2)

    lov = ['ut', 'un', 'ub', 'q', 'ef']
    res = simtools.Results(lot=pm.time_vector_output().tolist(), lov=lov, los=pm.los)
    res.update(0, s, lov, [ut, un, ub, q, ef])

    if wo.tbw is None:
        if fast:
            Is1 = I.multiply(1. + ht2)
            Is2 = I.multiply(1. / M)
            Is3 = I.multiply(ht * bt)
            Xb = np.zeros((3, n))
            Yb = np.zeros((3, n))
        else:
            A25 = I.multiply(M)
            A61 = I.multiply(wo.gm / w**2)
            A62 = I.multiply(wo.bt / w**2)
    else:
        if fast:
            Is1 = I.multiply(1. + ht2)
            Is2 = I.multiply(1. / M)
            Xb = np.zeros((3, n))
            Yb = np.zeros((3, n))
        else:
            A25 = I.multiply(M)

    # loop
    res.start_timer()
    pb = spb.generate(pm.pp, pm.nt, desc=__name__)
    for k in range(pm.nt):

        if wo.tbw is not None:
            u_ft = wo.tbw.get(t=t, y=0.5*cb.Lp)[0][0][0]
            #print("nt = %g, t = %gs u(t) = %gm/s" % (k, t, u_ft))
        
            w = 2. * np.pi * u_ft * wo.st / cb.d
    
            al = wo.al / w**2
            bt = wo.bt / w**2
            gm = wo.gm / w**2
        
            if fast:
                Is3 = I.multiply(ht * bt)
            else:
                A61 = I.multiply(wo.gm / w**2)
                A62 = I.multiply(wo.bt / w**2)

        h = -1. * cb.m * cb.g * cb.d / cb.H * un \
             + 0.5 * (cb.d / cb.L)**2 * ((C * un)**2 + (C * ub)**2)
        e = 0.5 * np.sum((h[:-1] + h[1:]) * ds)

        if fast:
            Aa = A.multiply((cb.H + cb.EA * e) / (cb.m * cb.L**2 * w**2) * ht)
            Bb = (cb.EA * cb.g * e) / (cb.H * cb.d * w**2)
            A21 = Aa
            A41 = Aa.multiply(al) + I.multiply(ht * gm)
            A44 = sp.sparse.diags([(-1. * ht * wo.eps) * (q[1:-1]**2 - 1.)], [0])
            R1 = un[1:-1] + ht * vn[1:-1]
            R2 = vn[1:-1] + A21 * un[1:-1] + (ht * M) * q[1:-1] + (dt * Bb) * J
            R3 = q[1:-1] + ht * r[1:-1]
            R4 = (r[1:-1] + A41 * un[1:-1] + (ht * bt) * vn[1:-1]
                  - ht * q[1:-1] + A44 * r[1:-1] + (dt * Bb * al) * J)
            if M == 0.:
                Xb[0, +1:] = -ht * A21.diagonal(k=1)
                Xb[1, :] = 1. - ht * A21.diagonal(k=0)
                Xb[2, :-1] = -ht * A21.diagonal(k=-1)
                Yy = Is1 - A44
                Yb[0, +1:] = Yy.diagonal(k=1)
                Yb[1, :] = Yy.diagonal(k=0)
                Yb[2, :-1] = Yy.diagonal(k=-1)
                vn[1:-1] = sp.linalg.solve_banded((1, 1), Xb, A21 * R1 + R2)
                un[1:-1] = ht * vn[1:-1] + R1
                r[1:-1] = sp.linalg.solve_banded((1, 1), Yb,
                                                 (R4 + A41 * un[1:-1]
                                                  + (ht * bt) * vn[1:-1] - ht * R3))
                q[1:-1] = ht * r[1:-1] + R3
            else:
                Xx = ((I - A44).multiply(1. / (M * ht2)) + Is2) * (I - A21.multiply(ht)) - A41.multiply(ht) - Is3
                Rs = (A21 * R1 + R2) / M
                Rx = R4 + A41 * R1 + Rs + (I - A44) * (Rs / ht2 + R3 / ht)
                Xb[0, +1:] = Xx.diagonal(k=1)
                Xb[1, :] = Xx.diagonal(k=0)
                Xb[2, :-1] = Xx.diagonal(k=-1)
                vn[1:-1] = sp.linalg.solve_banded((1, 1), Xb, Rx)
                un[1:-1] = ht * vn[1:-1] + R1
                q[1:-1] = (vn[1:-1] - A21 * un[1:-1] - R2) / (M * ht)
                r[1:-1] = (q[1:-1] - R3) / ht

        else:
            A21 = (cb.H + cb.EA * e) / (cb.m * cb.L**2 * w**2) * A
            A43 = A21
            A66 = sp.sparse.diags([q[1:-1]**2 - 1.], [0]).multiply(-1. * wo.eps)
            AA = sp.sparse.bmat([[None, I, None, None, None, None],
                                 [A21, None, None, None, A25, None],
                                 [None, None, None, I, None, None],
                                 [None, None, A43, None, None, None],
                                 [None, None, None, None, None, I],
                                 [A61 + A21.multiply(wo.al / w**2), A62, None, None,
                                  A25.multiply(wo.al / w**2) - I, A66]])
            X = np.concatenate((un[1:-1], vn[1:-1], ub[1:-1], vb[1:-1], q[1:-1], r[1:-1]))
            rhs = cb.EA * cb.g / (cb.H * cb.d * w**2) * e * np.ones(n)
            R = np.hstack((np.zeros(n), rhs, np.zeros(3 * n), wo.al / w**2 * rhs))
            
            Ma = sp.sparse.eye(6 * n) - AA.multiply(ht)
            Mb = sp.sparse.eye(6 * n) + AA.multiply(ht)
            
            # solve matrix problem
            Xn = sp.sparse.linalg.spsolve(Ma, Mb * X + dt * R)
            
            # recover next values into variable arrays
            un = np.concatenate(([0.], Xn[0 * n: 1 * n], [0.]))
            vn = np.concatenate(([0.], Xn[1 * n: 2 * n], [0.]))
            ub = np.concatenate(([0.], Xn[2 * n: 3 * n], [0.]))
            vb = np.concatenate(([0.], Xn[3 * n: 4 * n], [0.]))
            q = np.concatenate(([0.], Xn[4 * n: 5 * n], [0.]))
            r = np.concatenate(([0.], Xn[5 * n: 6 * n], [0.]))

        t += dt
        ut, ef = cbu.utef(un, ub, C, s, ds, vt2)
        if (k + 1) % pm.rr == 0:
            res.update((k // pm.rr) + 1, s, lov, [ut, un, ub, q, ef])
            pb.update(pm.rr)
    # END FOR
    pb.close()
    res.stop_timer()

    # add dim
    res.data.assign_coords({simtools.__stime__: res.data[simtools.__stime__].values * tAd})
    for v in ['ut', 'un', 'ub']:
        res.data[v] *= cb.d
    res.data['ef'] *= cb.EA
    
    return res


