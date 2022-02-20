#############################################################################
##                                                                         ##
##                  Mitaka signature scheme security estimator             ##
##                                                                         ##
#############################################################################

from math import sqrt, exp, log, pi, floor, log
import sys

## HARD CONSTANTS ##
e = exp(1)
q = 1024 * 12 + 1
NB_QUERIES = 2 ** 64 # NIST Recommendation

#############################################################################

def smooth(eps, n):
    ''' Estimation of the somoothing parameter of ZZ^n '''
    return sqrt(log(2*n*(1+1/eps))/pi)/sqrt(2*pi)

def print_security(B):
    '''
        Converts BIKZ security in bits for classical and quantum sieving
        using the core-svp methodology
    '''
    sec_qrec_classical  = floor(B*0.292)
    sec_qrec_quantum    = floor(B*0.265)
    print(" BIKZ:\t"+str(B))
    print(" Classical:\t"+str(sec_qrec_classical))
    print(" Quantum:\t"+str(sec_qrec_quantum))


def mitaka_security(n, sigma_offset, fg_norm, target_bitsec, \
        target_rejection = 0.1, verbose=True):
    '''
    Mitaka parameter estimation
            - n is the dimension of the convolution ring (512 or 1024)
            - sigma_offset is the accessible standard deviation offset
            using the Hybrid sampler
            - fg_norm is the L_2 norm of the vector (f,g) in the canonical
              embedding
            - target_rejection is the maximal acceptable rate of rejection
            for signatures
    '''
    # Computation of the effective deviation of the hybrid sampler
    eps = 1/sqrt(target_bitsec * NB_QUERIES)
    sigma = sigma_offset*sqrt(q)*smooth(eps, 2*n)

    # Estimate of the signature size w.r.t rejection probability
    tau = 1.1
    while (1):
        max_sig_norm = floor(tau*sqrt(2*n)*sigma)
        rejection_rate = exp(2*n*(1-tau**2)/2)*tau**(2*n)
        if rejection_rate > target_rejection:   break
        else:                                   tau-= 0.001

    # Key recovery
    B = 100 # Initial blocksize
    sigma_fg = fg_norm/sqrt(2*n)
    while (1):
        left = (B/(2*pi*e))**(1-n/B) * sqrt(q)
        right = sqrt(3 * B/4)*sigma_fg
        if left > right: break
        else:            B += 1
    if verbose:
        print(" -----[ Key Recovery ] -----")
        print_security(B)

    # Signature forgery
    B = 100
    def condition_LH(beta):
        return min([(((pi*B)**(1/B)*B/(2 * pi * e)) ** ((2*n-k)/(2*B-2))) *
            q**(n/(2*n-k)) for k in range(n)])

    while condition_LH(B) > max_sig_norm: B += 1
    sec_forgery_classical = (B*0.292)
    sec_forgery_quantum   = (B*0.265)
    if verbose:
        print(" -----[ Signature forgery ] -----")
        print_security(B)
    return(sec_forgery_classical, sec_forgery_quantum)

#############################################################################
#                           Script bootstrap
#############################################################################

print((512,mitaka_security(512, 2.04, 1.17*sqrt(q), 128, verbose=False)))
print((648,mitaka_security(648, 2.13, sqrt(2)*sqrt(q), 128, verbose=False)))
print((768,mitaka_security(768, 2.20, sqrt(2)*sqrt(q), 150, verbose=False)))
print((864,mitaka_security(864, 2.25, sqrt(2)*sqrt(q), 180, verbose=False)))
print((972,mitaka_security(972, 2.30, sqrt(2)*sqrt(q), 200, verbose=False)))
print((1024,mitaka_security(1024, 2.33, 1.17*sqrt(q), 200, verbose=False)))
print((512,mitaka_security(512, 1.17, 1.17*sqrt(q), 128, verbose=False)))
print((1024,mitaka_security(1024, 1.17, 1.17*sqrt(q), 256, verbose=False)))
print((512,mitaka_security(512, 3.03, 1.17*sqrt(q), 128, verbose=False)))
print((1024,mitaka_security(1024, 3.58, 1.17*sqrt(q), 128, verbose=False)))
