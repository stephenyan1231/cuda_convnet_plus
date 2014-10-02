

import numpy as np
from scipy import linalg as LA

class PCA:
    def __init__(self,data):
        # data: n * d array
        n=data.shape[0]
        self.mean = np.sum(data,0)/n
        cData=data-self.mean
        covMat=np.cov(cData,rowvar=0)
        evals,evecs=LA.eigh(covMat)
        evals=evals.real
        idx=np.argsort(evals)[::-1]
        self.evals=evals[idx]
        self.evecs=evecs[:,idx]
        self.cumSumEval=np.cumsum(self.evals)
        self.evalSum=np.sum(self.evals)
        self.cumFracEval=self.cumSumEval/self.evalSum
                
    def proj(self,data,varKeep=1.0):
        # input: data: n * d array
        # output: ret: n*d
        dimKeep=np.flatnonzero(self.cumFracEval>=varKeep)[0]
        ret = np.dot(data-self.mean,self.evecs[:,range(dimKeep+1)])
        return ret
    
    # ret: n*d
    def proj_topk(self,data,num_comp):
        ret = np.dot(data-self.mean,self.evecs[:,:num_comp])
        return ret
    
    def get_mean_evals_evecs_cumFracEval(self):
        return self.mean,self.evals,self.evecs,self.cumFracEval