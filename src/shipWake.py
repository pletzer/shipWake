#!/usr/bin/env python

import numpy as np
from scipy.integrate import quad_vec
import vtk

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
    
    
    def show(self, xx, yy, zz):

        ny1, nx1 = xx.shape
        numPoints = nx1 * ny1
        fMin = zz.min()
        fMax = zz.max()

        # create the pipeline objects
        data = vtk.vtkDoubleArray()
        coords = vtk.vtkDoubleArray()
        pts = vtk.vtkPoints()
        grid = vtk.vtkStructuredGrid()
        dataMapper = vtk.vtkDataSetMapper()
        dataActor = vtk.vtkActor()
        lut = vtk.vtkLookupTable()
        cbar = vtk.vtkScalarBarActor()

        # build the lookup table
        ncolors = 64
        lut.SetNumberOfColors(ncolors)
        for i in range(ncolors):
            x = 0.5*i*np.pi/(ncolors - 1.)
            r = np.sin(3*x)**2
            g = np.sin(1*x)**2
            b = np.sin(5*x)**2
            a = 1.0 # opacity
            lut.SetTableValue(i, r, g, b, a)
        lut.SetTableRange(fMin, fMax)
        cbar.SetLookupTable(lut)
        dataMapper.SetUseLookupTableScalarRange(1)

        # construct the grid
        grid.SetDimensions(nx1, ny1, 1)

        xyz = np.empty((numPoints, 3), float)
        xyz[:, 0] = xx.flat
        xyz[:, 1] = yy.flat
        xyz[:, 2] = zz.flat
        coords.SetNumberOfComponents(3)
        coords.SetNumberOfTuples(numPoints)
        coords.SetVoidArray(xyz, 3*numPoints, 1)

        # set the data
        data.SetNumberOfComponents(1) # scalar
        data.SetNumberOfTuples(numPoints)
        # set the data to some function of the coordinate
        save = 1
        data.SetVoidArray(zz, numPoints, save)


        # connect
        pts.SetNumberOfPoints(numPoints)
        pts.SetData(coords)
        grid.SetPoints(pts)
        grid.GetPointData().SetScalars(data)
        dataMapper.SetInputData(grid)
        dataActor.SetMapper(dataMapper)
        dataMapper.SetLookupTable(lut)

        # show
        ren = vtk.vtkRenderer()
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        # add the actors to the renderer, set the background and size
        ren.AddActor(dataActor)
        ren.AddActor(cbar)
        ren.SetBackground(0.5, 0.5, 0.5)
        renWin.SetSize(900, 600)
        iren.Initialize()
        renWin.Render()
        iren.Start()
       
            
    
if __name__ == '__main__':
        sw = ShipWake(nx=32, ny=16)
        x, y = -20., 1.2
        zeta = sw.compute(x=-20., y=1.2, froude=0.5)
        print(f'(x,y)=({x:.2f}, {y:.2f} zeta = {zeta}')
        
        x = np.linspace(-140, 10., 21)
        y = np.linspace(0., 50., 11)
        xx, yy = np.meshgrid(x, y)
        zz = sw.compute(x=xx, y=yy, froude=0.5)
        print(f'zz.real min/max = {zz.real.min()}/{zz.real.max()}')