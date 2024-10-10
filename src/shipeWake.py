#!/usr/bin/env python

import numpy as np
from scipy.integrate import quad_vec

class ShipWake(object):
    
    def __init__(self, elx=140.0, nx=128, ny=64) -> None:
        
        # gravitational constant 
        self.g = 9.81
        
        nx1 = nx + 1
        ny1 = ny + 1
        
        # grid
        self.x = np.linspace(-elx, 0., nx1)
        self.y = np.linspace(0., elx/2., ny1)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        
    
    def compute(self, x, y, froude=0.5, elShip=4.0, eps=1.e-10) -> None:
        
        u =  np.sqrt(self.g * elShip) * froude
        kg = self.g/u**2
        fourPiSquare = 4 * np.pi**2
        
        def integrand(k):
            d = np.sqrt(1.0 - kg/k)
            kx = np.sqrt(kg * k)
            ky = np.sqrt(k**2 - kx**2)
            return ( \
                np.exp( 1j * (kx*x + ky*y) ) \
                * (-1j * kx) * np.exp(-k**2 * elShip**2 / fourPiSquare)
                    ) / d
            
        def realIntegrand(k):
            return integrand(k).real
        
        def imagIntegrand(k):
            return integrand(k).imag
        
        return quad_vec(realIntegrand, kg + eps, np.inf)[0] + 1j*quad_vec(imagIntegrand, kg + eps, np.inf)[0]
            
    
if __name__ == '__main__':
        sw = ShipWake(nx=32, ny=16)
        x, y = -20., 1.2
        zeta = sw.compute(x=-20., y=1.2, froude=0.5)
        print(f'(x,y)=({x:.2f}, {y:.2f} zeta = {zeta}')