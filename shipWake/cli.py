import argparse
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
        print(f'u = {u:.2f} k_g = {kg:.2g} lambda_g = {2*np.pi/kg:.2f}')
        eightPiSquare = 8 * np.pi**2
        
        def integrand(k):
            d = np.sqrt(1.0 - kg/k)
            kx = np.sqrt(kg * k)
            ky = np.sqrt(k**2 - kx**2)
            coeff = 1j * np.sqrt(self.g) * np.sqrt(k) * elShip**2 / (u * eightPiSquare)
            return coeff \
                * np.exp( 1j * (kx*x + ky*y) ) \
                * np.exp(-k**2 * elShip**2 / eightPiSquare \
                    ) / d
            
        def realIntegrand(k):
            return integrand(k).real
        
        def imagIntegrand(k):
            return integrand(k).imag
        
        return quad_vec(realIntegrand, kg + eps, np.inf)[0]
    
    
    def show(self, xx, yy, zz):

        ny1, nx1 = xx.shape
        numPoints = nx1 * ny1
        fMin, fMax = zz.min(), zz.max()
        fMax = max(abs(fMax), abs(fMin))
        fMin = -fMax

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


def main():
    """
    Compute the wake field
    
    :param float froude: Froude number
    :param float elx: length of domain along the ship's path
    :param float elShip: length of the ship
    :param int nx: number of x cells
    :param int ny: number of y cells
    """
    parser = argparse.ArgumentParser(
                    description='Compute the wake behind a ship')

    parser.add_argument('-f', '--froude', type=float, default=0.5, help='Froude number')
    parser.add_argument('-x', '--xlen', type=float, default=140.0, help='Length of domain along path')
    parser.add_argument('-s', '--shiplen', type=float, default=4.0, help='Size of ship')
    parser.add_argument('-n', '--nx', type=int, default=128, help='Number of cells in x direction')
    parser.add_argument('-m', '--my', type=int, default=64, help='Number of cells in y direction')

    args = parser.parse_args()

    print(f'Fr={args.froude} ship len: {args.shiplen}')
    
    sw = ShipWake(elx=args.xlen, nx=args.nx, ny=args.my)
    
    buffer = 0.1 * args.shiplen
    x = np.linspace(-args.xlen + buffer, buffer, args.nx + 1)
    y = np.linspace(0.0, args.xlen/2.0, args.my + 1)
    xx, yy = np.meshgrid(x, y)
    zzr = sw.compute(xx, yy, froude=args.froude, elShip=args.shiplen)
    
    print(f'mean, std height: {zzr.mean()}, {zzr.std()}')

    sw.show(xx, yy, zzr)

if __name__ == '__main__':
    main()