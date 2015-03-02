#This is code for the phenome_model inference
from __future__ import division; import cython, pylab;pylab.ion(); import numpy as np; import scipy.stats as stats; import time; import scipy.special as special; import cPickle as pickle
import pyximport; pyximport.install(); import samplers as ss

# Load vocabularies
with open('../data/examples/diag_vocab.txt') as f:
    icd9_vocab = np.array(f.read().split('\n'))
with open('../data/examples/word_vocab.txt') as f:
    term_vocab = np.array(f.read().split('\n'))
with open('../data/examples/med_vocab.txt') as f:
    med_vocab = np.array(f.read().split('\n'))
with open('../data/examples/lab_vocab.txt') as f:
    lab_vocab = np.array(f.read().split('\n'))

######## Set constants ##############  MOVE THESE TO ARGUMENTS
# Number of phenotypes
P=250
num_iterations=7000
#Priors and initializations
card_I = len(icd9_vocab); card_N = len(term_vocab); card_O = len(med_vocab); card_M = L = len(lab_vocab) 
alpha= 0.1; mu = 0.1; nu = 0.1; xi = 0.1; pi = 0.1
vs = []; ws = []; xs = []; ys = []

with open('../data/examples/diag_counts.txt') as f:
    for i, line in enumerate(f):
        data = [int(x.split(':')[0]) for x in line.strip('\n').split(',')[1:]] #count of the icd9 is always 1, we don't need the count, and first one is the hadmid
        vs.append(np.array(data,np.int))

with open('../data/examples/word_counts.txt') as f:
    for i, line in enumerate(f):
        data = [x for x in line.strip('\n').split(',')[1:]]
        lodata=[]
        for each in data:
            for i in range(int(each.split(':')[1])):
                lodata.append(int(each.split(':')[0]))
        ws.append(np.array(lodata,np.int))
    
with open('../data/examples/med_counts.txt') as f:
    for i, line in enumerate(f):
        data = [x for x in line.strip('\n').split(',')[1:]]
        lodata=[]
        for each in data:
            for i in range(int(each.split(':')[1])):
                lodata.append(int(each.split(':')[0]))
        xs.append(np.array(lodata,np.int))

with open('../data/examples/lab_counts.txt') as f:
    for i, line in enumerate(f):
        data=[x for x in line.strip('\n').split(',')[1:]]
        lodata=[]
        for each in data:
            lab_id=each.split(':')[0]
            for i in range(len(each.split(';'))):
                lodata.append(int(lab_id))
        ys.append(np.array(lodata,np.int))
        
# Number of Records
R = len(ys)

# Initialize latent variables
gammas = [np.random.randint(P,size=x) for x in [len(y) for y in vs]]
deltas = [np.random.randint(P,size=x) for x in [len(y) for y in ws]]
epsilons = [np.random.randint(P,size=x) for x in [len(y) for y in xs]]
zetas = [np.random.randint(P,size=x) for x in [len(y) for y in ys]]

#initialize the count variables
def init_vars(passign,dtype,obs):
    for i,record in enumerate(passign):
        for assi in range(len(record)):
            phenotype=record[assi]
            val=obs[i][assi]
            dtype[val,phenotype]+=1
            record_counts[i,phenotype]+=1

record_counts=np.zeros((R,P))
diag_counts=np.zeros((card_I,P))
init_vars(gammas,diag_counts,vs)
diag_pheno=np.sum(diag_counts,0)

doc_counts=np.zeros((card_N,P))
init_vars(deltas,doc_counts,ws)
doc_pheno=np.sum(doc_counts,0)

ord_counts=np.zeros((card_O,P))
init_vars(epsilons,ord_counts,xs)
ord_pheno=np.sum(ord_counts,0)

lab_counts=np.zeros((card_M,P))
init_vars(zetas,lab_counts,ys)
lab_pheno=np.sum(lab_counts,0)

#Define the collapsed log-likelihood
def c_joint_ll(diag_counts,doc_counts,ord_counts,lab_counts):
    ll=0
    ll+=special.gammaln(alpha*P)-P*special.gammaln(alpha)
    ll+=np.sum(special.gammaln(alpha+record_counts))-np.sum(special.gammaln(np.sum((record_counts+alpha),1)))
    ll+=P*(special.gammaln(mu*card_I)-card_I*special.gammaln(mu)+
        special.gammaln(nu*card_N)-card_N*special.gammaln(nu)+
        special.gammaln(xi*card_O)-card_O*special.gammaln(xi)+
        special.gammaln(pi*card_M)-card_M*special.gammaln(pi))
    ll+=np.sum(special.gammaln(mu+diag_counts))-np.sum(special.gammaln(mu*card_I+np.sum(diag_counts,0)))
    ll+=np.sum(special.gammaln(nu+doc_counts))-np.sum(special.gammaln(nu*card_N+np.sum(doc_counts,0)))
    ll+=np.sum(special.gammaln(xi+ord_counts))-np.sum(special.gammaln(xi*card_O+np.sum(ord_counts,0)))
    ll+=np.sum(special.gammaln(pi+lab_counts))-np.sum(special.gammaln(pi*card_M+np.sum(lab_counts,0)))
    return ll

# Sample all at once for now
ll_trail = []

p_for_cy=np.zeros(P)
for iteration in range(0,num_iteration):
    print iteration
    starttime=time.time()
    for rec_i in range(R):
        rcounts=record_counts[rec_i,:]
        grand_num=np.random.rand(len(gammas[rec_i]))
        ss.sample_assign(P,alpha,gammas[rec_i],vs[rec_i],diag_counts,diag_pheno,mu,rcounts,grand_num,p_for_cy)
        drand_num=np.random.rand(len(deltas[rec_i]))
        ss.sample_assign(P,alpha,deltas[rec_i],ws[rec_i],doc_counts,doc_pheno,nu,rcounts,drand_num,p_for_cy)
        erand_num=np.random.rand(len(epsilons[rec_i]))
        ss.sample_assign(P,alpha,epsilons[rec_i],xs[rec_i],ord_counts,ord_pheno,xi,rcounts,erand_num,p_for_cy)
        zrand_num=np.random.rand(len(zetas[rec_i]))
        ss.sample_assign(P,alpha,zetas[rec_i],ys[rec_i],lab_counts,lab_pheno,pi,rcounts,zrand_num,p_for_cy)
    print "done sampling", (time.time()-starttime)/60.
    if iteration%1000==0:
        save_assignments(iteration)
    # Pick a random phenotype
    if iteration%5==0:
        ll_trail.append(c_joint_ll(diag_counts,doc_counts,ord_counts,lab_counts))
        print ll_trail[-1],ll_trail
        phen_pick = np.random.randint(P)
        print "Phenotype", phen_pick
        # Print properties
        sort = np.argsort(diag_counts[:,phen_pick])[::-1]
        print "Diagnoses",icd9_vocab[sort[:20]]
        sort = np.argsort(doc_counts[:,phen_pick])[::-1]
        print "Terms",term_vocab[sort[:20]]
        sort = np.argsort(ord_counts[:,phen_pick])[::-1]
        print "Meds",med_vocab[sort[:20]]
        sort = np.argsort(lab_counts[:,phen_pick])[::-1]
        print "Labs",lab_vocab[sort[:20]]
        with open('../ll_trailfast_'+str(P)+'.data', 'a') as f:
            f.write(str(ll_trail[-1])+'\n')
        if iteration>1000:
            if float(ll_trail[-1])>=float(max(ll_trail)):
                save_assignments(str(iteration)+'max')
                with open('../ll_maxfast_'+str(P)+str(iteration)+'.txt','w') as g:
                    g.write(str(iteration)+'\n'+str(ll_trail[-1])+'\n')
                    for maxptype in range(P):
                        g.write("\nPHENOTYPE "+str(maxptype)+'\n')
                        dsort = np.argsort(diag_counts[:,maxptype])[::-1]
                        g.write("DIAGNOSES\n")
                        g.write(','.join(icd9_vocab[dsort[:20]])+'\n')
                        tsort = np.argsort(doc_counts[:,maxptype])[::-1]
                        g.write("TERMS\n")
                        g.write(','.join(term_vocab[tsort[:20]])+'\n')
                        msort = np.argsort(ord_counts[:,maxptype])[::-1]
                        g.write("MEDS\n")
                        g.write(','.join(med_vocab[msort[:20]])+'\n')
                        lsort = np.argsort(lab_counts[:,maxptype])[::-1]
                        g.write("LABS\n")
                        g.write(','.join(lab_vocab[lsort[:20]])+'\n')
