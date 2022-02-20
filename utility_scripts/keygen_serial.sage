from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler

import numpy as np
import itertools
from datetime import datetime

params = {  512: {'mK': 1024, 'thr': 2.04, 'q': 12289},
            648: {'mK': 1944, 'thr': 2.13, 'q': 3889 },
            768: {'mK': 2304, 'thr': 2.20, 'q': 18433},
            864: {'mK': 2592, 'thr': 2.25, 'q': 10369},
            972: {'mK': 2916, 'thr': 2.30, 'q': 17497},
           1024: {'mK': 2048, 'thr': 2.33, 'q': 12289} }

param = params[512]

mK  = param['mK']
thr0= param['thr']
q   = param['q']

d = euler_phi(mK)
zmstar = [a for a in range(mK) if gcd(a,mK)==1]
K.<z> = CyclotomicField(mK) # WARNING: degree(K) = d

def permmK(a):
    def rank(xs):
        # double argsort trick
        return xs.argsort().argsort()
    assert(gcd(a,mK)==1)
    return rank(np.array([(a*x)%mK for x in zmstar]))

gc = np.vstack([permmK(a) for a in zmstar[:d//2]])

def selecttest(m,thr=thr0,sigratio=1.17,trials=50):
    l=[]
    for i in range(trials):
        now = datetime.now()
        print("%s: test %d/%d" % (now.strftime("%H:%M:%S"),i+1,trials))
        l.append(selectfg(sigratio=sigratio,m=m,thr=thr)[0])
    return sorted(l)

def Kfft(x):
    return np.fft.fft(vector(RDF,x.list()),mK)[zmstar]

def qual(f,g):
    z = np.absolute(Kfft(f))^2 + np.absolute(Kfft(g))^2
    return sqrt(max(np.max(z)/q, q/np.min(z)))

def qualfft(f,g):
    z = np.absolute(f)^2 + np.absolute(g)^2
    return sqrt(max(np.max(z)/q, q/np.min(z)))

def selectfg(sigratio=1.17,m=16,ng=d//2,thr=thr0,verbose=False,early=False,alg='qf',qford=25):
    sigma0 = RDF(sigratio*sqrt(q/(4*d)))

    eps = 1/(4*sqrt(256*2^64))
    smoothing = 1/pi * sqrt(log(2*d*(1+1/eps))/2)
    if verbose:
        print("sigma_0 = %f" % sigma0)
        print("eta_eps = %f" % smoothing)

    D0 = DiscreteGaussianDistributionIntegerSampler(sigma=sigma0)

    if verbose:
        print("Generating %d Gaussian samples" % (4*m))
    fs = [ K( [ D0() for _ in range(d) ] ) for _ in range(2*m) ]
    gs = [ K( [ D0() for _ in range(d) ] ) for _ in range(2*m) ]
    if verbose:
        print("Computing the embeddings")
    fs_fft = np.array([Kfft(f) for f in fs])
    gs_fft = np.array([Kfft(g) for g in gs])
    if verbose:
        print("Computing the pairwise sums")
    Lf = (fs_fft[:m].reshape((m,1,-1)) + \
            fs_fft[m:].reshape((1,m,-1))).reshape((m^2,-1))
    Lg = (gs_fft[:m].reshape((m,1,-1)) + \
            gs_fft[m:].reshape((1,m,-1))).reshape((m^2,-1))

    if verbose:
        print("Computing the relative norms and filtering")
    Lu = np.real(np.multiply(Lf, np.conjugate(Lf)))
    Lv = np.real(np.multiply(Lg, np.conjugate(Lg)))

    indu = np.nonzero(np.amax(Lu,1) < thr^2*q)[0]
    indv = np.nonzero(np.amax(Lv,1) < thr^2*q)[0]
    Lu = Lu[indu]
    Lv = Lv[indv]

    if verbose:
        print("Expanding via Galois action")
    Lv = np.vstack([ Lv[:,g] for g in gc[:ng] ])

    #discard redundant second half
    Lu = Lu[:,:d//2]
    Lv = Lv[:,:d//2]

    if verbose:
        print("List sizes: |Lu| = %d, |Lv| = %d" % (len(Lu), len(Lv)))

    if verbose:
        print("Searching for min max embedding")
    r = None
    sinf = q/thr^2
    kept = 0
    for j, u in enumerate(Lu):
        if alg is 'qf':
            uord = np.argpartition(u,qford)[:qford]
            zqf = np.amin(Lv[:,uord] + u[uord],1)
            w = np.nonzero(zqf >= sinf)[0]
            if len(w) == 0:
                continue
            kept += len(w)
            z  = Lv[w,:] + u
        else:
            z  = Lv + u
        z1 = np.amax(z,1)/q
        z0 = q/np.amin(z,1)
        zm = np.maximum(z0,z1)
        am = np.argmin(zm)
        sz = sqrt(zm[am])

        if r is None or sz < r:
            r  = sz
            xu = j
            if alg is 'qf':
                yu = w[am]
            else:
                yu = am
        if early and sz < thr:
            break

    if r is None:
        if verbose:
            print(">>> Not found <<< ")
        raise RuntimeError("No candidate key found")

    if verbose and alg is 'qf':
        print("Kept candidates for (u,v): %d/%d (%f%%)" % \
            (kept,len(Lu)*len(Lv),float(100*kept/len(Lu)/len(Lv))))
    
    f_ind = indu[xu]
    g_ind = indv[yu % len(indv)]
    t_ind = yu // len(indv)
    f0 = fs[f_ind//m] + fs[(f_ind%m) + m]
    g0 = gs[g_ind//m] + gs[(g_ind%m) + m]
    if verbose:
        print(">>> Best candidate: %f <<<" % r)
    tau = [(Integers(mK)(x)/zmstar[t_ind]).lift() for x in range(mK)]
    g0l = g0.list() + [0]*(mK-d)
    g0conj = K(list(np.array(g0l)[tau]))
    return r,f0,g0conj

def computeFG(f,g):
    # outputs F,G such that fG-gF = q and (f,g), (F,G) generates the
    #  orthogonal lattice of (f/g mod q, -1) as long as g is invertible mod q
    Nf = f.norm()
    Ng = g.norm()
    s = Nf/f
    t = Ng/g

    d,u,v = xgcd(Nf,Ng) # u*Nf + v*Nd = d in ZZ
    F_big = -q*t*v/d
    G_big = q*s*u/d

    assert f*G_big-g*F_big == q
    ktemp = (f.conjugate()*F_big+g.conjugate()*G_big)/(f*f.conjugate() + g*g.conjugate())
    k = K( [ e.round() for e in ktemp.list() ] )

    F = F_big - k*f
    G = G_big - k*g

    assert f*G-g*F == q
    return (F,G)

def keygen(sigratio=1.17,m=16,ng=d//2,thr=thr0,verbose=False,early=False,alg='qf'):
    itfg = 0
    if verbose:
        print("Generating (f,g) pair.")
    while True:
        itfg += 1
        if verbose:
            print("...iteration %d" % itfg)
        try:
            s,f,g = selectfg(sigratio,m,ng,thr,False,early,alg)
            if s <= thr:
                if verbose:
                    print("...success!")
                break
            else:
                if verbose:
                    print("...failed (s = %f > %f)" % (s,thr))
        except RuntimeError:
            if verbose:
                print("...failed (not found)")
            pass

    if verbose:
        print("Generating (F,G) pair.")
    F,G = computeFG(f,g)
    if verbose:
        print("Key generation complete!")
    return f,g,F,G

# vim: ft=python ts=4 expandtab
