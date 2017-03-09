from sklearn import mixture;
import numpy;
import copy;
#import pdb;

class GmmMap(mixture.GMM):
    relevance_factor = 16;
    
    def map_adapt(self, x):
        gmmCurrent = copy.deepcopy(self);
        logLikelihoodOld = gmmCurrent.score(x).sum();
        nIters = self.n_iter;
        nObs = x.shape[0];
        nDim = x.shape[1];
        self.converged_ = False;
        while (nIters > 0):
            post = gmmCurrent.predict_proba(x);
            post_sum = numpy.sum(post, 0);
            mus = numpy.zeros((self.n_components, nDim));
            sigmas = numpy.zeros((self.n_components, nDim));
            for idxGauss in range(self.n_components):
                for idxFrame in range(nObs):
                    mus[idxGauss,:] = mus[idxGauss,:] + post[idxFrame,idxGauss] * x[idxFrame,:];
                    sigmas[idxGauss,:] = sigmas[idxGauss,:] + post[idxFrame,idxGauss] * x[idxFrame,:]**2;
                
                #check if there is a zero value the produces a zere division-> nan value: mu_hat=alpha*mus+...=0*nan+...=nan+.... 
                #if yes set the mus and sigmas to zero for non adaptation.
                if(post_sum[idxGauss]==0): 
                    mus[idxGauss,:]=0;
                    sigmas[idxGauss,:] =0;
                else:
                    mus[idxGauss,:] = mus[idxGauss,:] / post_sum[idxGauss];
                    sigmas[idxGauss,:] = sigmas[idxGauss,:] / post_sum[idxGauss];


            
            alpha = post_sum / (post_sum + self.relevance_factor);
            if ("w" in self.params):
                weights_hat = alpha * post_sum / nObs + (1-alpha) * gmmCurrent.weights_;
                weights_hat = weights_hat / weights_hat.sum();
            else:
                weights_hat = gmmCurrent.weights_;
            
            mu_hat     = numpy.zeros((self.n_components, nDim));
            sigmas_hat = numpy.zeros((self.n_components, nDim));
            for idxGauss in range(self.n_components):
                if ("m" in self.params or "c" in self.params):
                    mu_hat[idxGauss,:] = alpha[idxGauss] * mus[idxGauss,:] + (1-alpha[idxGauss]) * gmmCurrent.means_[idxGauss,:];
                else:
                    mu_hat[idxGauss,:] = gmmCurrent.means_[idxGauss,:];
                
                if ("c" in self.params):
                    sigmas_hat[idxGauss,:] = alpha[idxGauss] * sigmas[idxGauss,:] + (1-alpha[idxGauss]) * (gmmCurrent.covars_[idxGauss,:] + gmmCurrent.means_[idxGauss,:]**2)-mu_hat[idxGauss,:]**2;
                else:
                    sigmas_hat[idxGauss,:] = gmmCurrent.covars_[idxGauss,:];
        
            gmmCurrent.means_  = mu_hat;
            gmmCurrent.covars_ = sigmas_hat;
            gmmCurrent.weights_= weights_hat;
            
            logLikelihoodCurr = gmmCurrent.score(x).sum();
            logLikelihoodDelta = logLikelihoodCurr - logLikelihoodOld;
            logLikelihoodOld = logLikelihoodCurr;
            
#             print(logLikelihoodDelta/abs(logLikelihoodCurr));
            if (logLikelihoodDelta < (0.001 * abs(logLikelihoodCurr))):
                self.converged_ = True;
                #print(logLikelihoodDelta/abs(logLikelihoodCurr));
                break;
            
            nIters = nIters - 1;
        
#         print(self.n_iter - nIters);
        self.means_  = copy.deepcopy(gmmCurrent.means_);
        self.covars_ = copy.deepcopy(gmmCurrent.covars_);
        self.weights_= copy.deepcopy(gmmCurrent.weights_);
#         print(self.score(x).sum());
                    
