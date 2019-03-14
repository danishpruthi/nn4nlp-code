import numpy as np
from tqdm import tqdm
from numpy.random import multivariate_normal
from numpy.random import uniform
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import poisson

# useful constants
TAU_0 = 0.0001
TAU_1 = 0.0001
ALPHA = 0.1
BETA = 0.1
ATTACK_IDX_0 = 0
DEF_IDX_0 = 20

# tune-able params
t = 1
sigma = 0.005


# Y is observed
Y = np.genfromtxt('premier_league_2013_2014.dat', delimiter=',')


'''
    function to compute the joint log prob
    Y is observed, 380*2 goal data
    theta = {atk-20, def-20, home}
    eta is 4 dim set {mu_atk, mu_def, tau_atk, tau_def}
    total 45 params, atk-0, def-0 are fixed to 0.
'''
def log_P_joint(Y, theta, eta):
    mu_attack = eta[0]
    mu_def = eta[1]
    tao_attack = eta[2]
    tao_def = eta[3]
    h_attacks = theta[:20]
    h_defs= theta[20:40]
    home = theta[40]

    # log P(eta) computed below
    log_p = np.log(norm.pdf(mu_attack, 0, 1/TAU_1))
    log_p += np.log(norm.pdf(mu_def, 0, 1/TAU_1))
    log_p + np.log(gamma.pdf(tao_attack, a=ALPHA, scale=1/BETA))
    log_p + np.log(gamma.pdf(tao_def, a=ALPHA, scale=1/BETA))

    # log P(theta | home) computed below
    log_p += np.log(norm.pdf(home, 0, 1/TAU_0))

    # log P(h_attacks | eta)
    for h_attack in h_attacks:
        log_p + np.log(norm.pdf(h_attack, mu_attack, 1/tao_attack))

    # log P(h_defs | eta)
    for h_def in h_defs:
        log_p + np.log(norm.pdf(h_def, mu_def, 1/tao_def))

    # log P(Y|theta, eta)
    for g1, g2, t1, t2 in Y: # score (g1 : g2) in team t1 vs t2
        t1, t2 = int(t1), int(t2)
        poisson_param_1 = np.exp(home + h_attacks[t1] - h_defs[t2])
        poisson_param_2 = np.exp(h_attacks[t2] - h_defs[t1])
        log_p += poisson.logpmf(g1, mu=poisson_param_1)
        log_p += poisson.logpmf(g2, mu=poisson_param_2)


    return log_p


def generate_candidate(x, sigma):
    cov = sigma*sigma*np.eye(x.shape[0], x.shape[0])
    x_ = multivariate_normal(x, cov)

    # the variances can't be negative. If they are, resample
    while x_[43] < 0 or x_[44] < 0:
        x_ = multivariate_normal(x, cov)

    # because of the grounding to zero, do not sample two sacred indices
    x_[ATTACK_IDX_0] = 0
    x_[DEF_IDX_0] = 0

    return x_


x = 0.1 * np.ones(45)
x[ATTACK_IDX_0] = 0
x[DEF_IDX_0] = 0
samples = []
samples_rejected = 0.0

for ITER in tqdm(range(5000 + 5000*t)):
    x_ = generate_candidate(x, sigma)

    log_p_x_ = log_P_joint(Y, x_[:41], x_[41:])
    log_p_x = log_P_joint(Y, x[:41], x[41:])

    a_prob = min(1.0, np.exp(log_p_x_ - log_p_x))

    u = uniform()

    if a_prob >= u: # accept the dude
        x = x_
    else:
        samples_rejected += 1.0

    # collect samples...
    if ITER > 5000 and ITER%t==0: # after burn in
        samples.append(x)

    print ("Rejection Ratio = %0.2f \r" %(samples_rejected/(ITER+1)))