import numpy as np
import os
from astropy.io import fits
import shutil
import random
import warnings
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter('ignore', category=VerifyWarning)
warnings.simplefilter('ignore', category=ValueError)
warnings.simplefilter('ignore', category=RuntimeWarning)

#floor magnitude:
floor = 30
class Smac_Map():
    def __init__(self, image_path):
        self.image_path = image_path
        self.name = os.path.basename(os.path.normpath(image_path))
        with fits.open(self.image_path) as hdu:
            hdu.verify('fix')
            self.data = hdu[1].data
    
    def get_size(self):
        '''
        Returns the size of the map size
        '''
        return int(self.data.size**(1./2))


class Transformation_Map(Smac_Map):
    
    def get_data(self):
        return self.data

    def apply_perturbations(self, perturbation, *args):
        '''
        This function applies one transformation to the map.         
        '''       
        #possible perturbations
        perturb_functions = {"resize":self.resize_image, "ellipse": self.draw_ellipse, "rotate" : self.rotate_map}       
        perturb_functions[perturbation](*args)
        return self

    def to_magnitude(self):
        '''
        Transform matrix of luminosities in magnitudes. Floor level 30
        '''
        self.data = -2.5 * np.log10(self.data)
        self.data[np.isinf(self.data)] = floor
        return self
    
    def to_luminosity(self):
        '''
        Transform matrix of luminosities in magnitudes. Floor level 30
        '''
        self.data = 10**(-self.data/2.5)
        
        self.data[self.data == 10**(-floor/2.5)] = 0.
        return self
    
    
    def resize_image(self, crop=[0,0,0,0]):
        '''
        Change size of the image
        
        +++++++++++++++++ PARAMETERS ++++++++++++++++++++++
        :: crop :: (list) in the following order (T, R, B, L) the image is cropped.
        If the cropping is random, use function random_size()
        '''
        size = self.get_size()
        t, r, b, l = crop
        self.data = self.data[t:size-b,l:size-r] 
        
        return self

    
    def draw_ellipse(self,maschera):
        '''
        Apply the masking from the ellipses
        '''
        masked_data = np.ma.masked_array(self.data, mask=maschera).filled(fill_value=0)
        self.data = masked_data
        return self
    
    
    def rotate_map(self, types):
        '''
        This function actually applies the transformation
        '''
        if types == 0:
            return self
        elif types == 1:
            self.data = np.flip(self.data)
            return self
        elif types == 2:
            self.data = np.flip(self.data,axis=1)
            return self
        elif types == 3:
            self.data = np.rot90(self.data)
            return self
        elif types == 4:
            self.data = np.transpose(self.data)
            return self


        
def random_size(size, pixels = 512):
    '''
    Returns possible cropping edges to get image of given pixels
    '''
    #top bottom
    a = random.randint(0, size-pixels)
    b = size - (pixels + a)
    
    #left right
    c = random.randint(0, size-pixels)
    d = size - (pixels + c)
    
    return [a,c,b,d]
    
def ellipse_equation(x,y,h,k,a,b,A):
    '''
    This function draws ellipses on the map. A is in radians
    '''
    first = (((x-h)*np.cos(A)-(y-k)*np.sin(A))**2)/a**2
    second = (((x-h)*np.sin(A)+(y-k)*np.cos(A))**2)/b**2
    return first+second    
    
def get_ellipse_parameters(n, factor = 0.7):  
    h = random.uniform(-1,1)
    k = random.uniform(-1,1)
    A = random.uniform(0,2*np.pi)

    a = random.betavariate(2,5) * factor
    b = a*random.uniform(0.3,.85)
    return h, k, a, b, A

def get_mask_ellipse(maps, N, n):
    '''
    This function draws random ellipses in map.
    '''
    # example: N = random.randint(2,10); n = random.randint(40,100)
    maschera = np.zeros_like(maps.data)
    masked_data = np.zeros_like(maps.data)
    binx = np.linspace(-1,1,maps.get_size())
    biny = np.linspace(-1,1,maps.get_size())


    print ("Large ellipses in the image are : ", N)
    for NN in range(N):
        par = get_ellipse_parameters(N, factor = 0.7)
        for cc, x in enumerate(binx):
            maschera[cc,np.where(ellipse_equation(x,biny,*par)<=1)[0]] = 1

    print ("The number of small ellipses: " , n)
    for NN in range(n):
        par = get_ellipse_parameters(n, factor = 0.15)
        for cc, x in enumerate(binx):
            maschera[cc,np.where(ellipse_equation(x,binx,*par)<=1)[0]] = 1
    return maschera