import numpy as np 
import healpy as hp 
import scipy.integrate as integrate
from tqdm import trange
from scipy.special import eval_legendre



class Needlet:

    def __init__(self, j, B, lmax):
        self.j = j
        self.B = B
        self.lmax = lmax

        self.filter = needlet_filter(self.j, self.B, self.lmax)
        self. b = self.filter.standardneedlet()
        self.ells = np.arange(lmax)

        self.NSIDE_X = get_nside(self.B, self.j)
        self.LAMBDA_j = 4*np.pi/hp.nside2npix(self.NSIDE_X)

    def compute_needlet(self, nside, eps_ind):
        ''' 
        nside: int, power of 2 
            Nside of the output needlet map

        eps_ind: int
            k index of eps_jk
        '''
    
        theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        vec_matrix = hp.ang2vec(theta, phi)

        eps_ang = hp.pix2ang(self.NSIDE_X, eps_ind)
        eps_vector = hp.ang2vec(self.NSIDE_X, eps_ang[0], eps_ang[1])

        dot_prod_arr = np.matmul(vec_matrix, eps_vector)

          
        legendre_arr = []
        for i in trange(self.lmax, desc = "Computing needlet map"):
            legendre_arr.append(eval_legendre(self.ells[i], dot_prod_arr))


        legendre_arr = np.array(legendre_arr)
        norm_arr = (self.ells*2 + 1)/(4*np.pi)
        needlet =  ( np.sqrt(self.LAMBDA_j)*
            np.sum(legendre_arr*norm_arr[:,None]*self.b[:-1][:,None], axis = 0) )
        return needlet

    
    def compute_needlet_coeffcients(self,input_map, nside_out = None):
        '''nside_out: Nonetype
            If not None, will give needlet coefficients map in nside_out 
            resoltution.'''

        if nside_out is not None:
            nside = nside_out
        else:
            nside = self.NSIDE_X
        
        alm = hp.map2alm(input_map)
        filtered_alm = np.sqrt(self.LAMBDA_j)*hp.almxfl(alm,self.b)
        beta = hp.alm2map(filtered_alm,nside, verbose=False)

        return beta



class needlet_filter:
    '''Adapted from https://github.com/javicarron/mtneedlet'''

    def __init__(self, j, B, lmax):
        self.j = j
        self. B = B
        self.lmax = lmax


        
    def __f_need(self,t):
        '''Auxiliar function f to define the standard needlet'''
        if t <= -1.:
            return(0.)
        elif t >= 1.:
            return(0.)
        else:
            return(np.exp(1./(t**2.-1.)))
        

    def __psi(self,u):
        '''Auxiliar function psi to define the standard needlet'''
        return(integrate.quad(self.__f_need,-1,u)[0]/integrate.quad(self.__f_need,-1,1)[0])

    def __phi(self,q):
        '''Auxiliar function phi to define the standard needlet'''
        B=float(self.B)
        if q < 0.:
            raise ValueError('The multipole should be a non-negative value')
        elif q <= 1./B:
            return(1.)
        elif q >= 1.:
            return(0)
        else:
            return(self.__psi(1.-(2.*B/(B-1.)*(q-1./B))))
        
    def __b2_need(self,xi):
        '''Auxiliar function b^2 to define the standard needlet'''
        b2=self.__phi(xi/self.B)-self.__phi(xi)
        return(np.max([0.,b2])) 
        ## np.max in order to avoid negative roots due to precision errors

    def standardneedlet(self):
        '''Return the needlet filter b(l) for a standard needlet with parameters 
        ``B`` and ``j``.
        
        Parameters
        ----------
        B : float
            The parameter B of the needlet, should be larger that 1.
        j : int or np.ndarray
            The frequency j of the needlet. Can be an array with the values of 
            ``j`` to be calculated.
        lmax : int
            The maximum value of the multipole l for which the filter will be 
            calculated (included).

        Returns
        -------
        np.ndarray
            A numpy array containing the values of a needlet filter b(l), 
            starting with l=0. If ``j`` is an array, it returns an array 
            containing the filters for each frequency.
            
        Note
        ----
        Standard needlets are always normalised by construction: the sum 
        (in frequencies ``j``) of the squares of the filters will be 1 for all 
        multipole ell.
        '''
        ls=np.arange(self.lmax+1)
        j=np.array(self.j,ndmin=1)
        needs=[]
        bl2=np.vectorize(self.__b2_need)

        for jj in j:
            xi=(ls/self.B**jj)
            bl=np.sqrt(bl2(xi))
            needs.append(bl)
            
        return(np.squeeze(needs))


def get_jmax(B, lmax):
    jmax = 0
    while(np.ceil(B**(jmax-1)) <lmax):
        jmax +=1
        
    return jmax -1

def get_nside(B, j):
    n = 1
    RHS = 0.5*np.floor(B**(j+1))
    while( 2**n < RHS ):
        n+=1
    return 2**n