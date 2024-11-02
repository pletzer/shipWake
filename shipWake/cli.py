import argparse
import numpy as np
from scipy.integrate import quad_vec
from scipy.interpolate import RegularGridInterpolator
import vtk

class FourierTransform(object):
    
    def __init__(self, xy, depths):
        
        # number points in x and y
        ns = len(xy[0]), len(xy[1])
        
        # domain length in x and y
        els = xy[0][-1] - xy[0][0], xy[1][-1] - xy[1][0]
        
        # lower x and y corners
        x0s = xy[0][0], xy[1][0]
        
        # intervals in x and y
        dxs = els[0]/ns[0], els[1]/ns[1]
        
        # Fourier grid
        ks = np.fft.fftshift(np.fft.fftfreq(ns[0])) * 2 * np.pi / dxs[0], \
             np.fft.fftshift(np.fft.fftfreq(ns[1])) * 2 * np.pi / dxs[1]
             
        kkx, kky = np.meshgrid(ks[0], ks[1])
                
        # Fourier transform, takes into account translation and rescaling. Also 
        # reorder the FFT terms to have continuously growing k values
        depthsFFT = np.exp(-1j*(kkx*x0s[0] + kky*x0s[1])) * dxs[0] * dxs[1] * \
            np.fft.fftshift(np.fft.fft2(depths))
            
        # extend Fourier domain to infinity
        kx_ext, ky_ext = np.empty((ns[0] + 4,), float), np.empty((ns[1] + 4,), float)
        kx_ext[2:-2] = ks[0]
        ky_ext[2:-2] = ks[1]
        kx_ext[0], kx_ext[1], kx_ext[-2], kx_ext[-1] = -1.e10, 2*ks[0][0], 2*ks[0][-1], +1.e10
        ky_ext[0], ky_ext[1], ky_ext[-2], ky_ext[-1] = -1.e10, 2*ks[1][0], 2*ks[1][-1], +1.e10
        
        # no ship hull perturbations for high k wavenumbers
        depthsFFT_ext = np.zeros((depthsFFT.shape[0] + 4, depthsFFT.shape[0] + 4), np.complex128)
        depthsFFT_ext[2:-2, 2:-2] = -depthsFFT    
            
        # build the interpolator
        self.interp = RegularGridInterpolator((kx_ext, ky_ext), depthsFFT_ext, method='linear')
    
    def __call__(self, k):
        return self.interp(k)

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
        
    
    def compute(self, x, y, xs, ys, depths, froude=0.5, eps=1.e-10) -> None:
        
        shipDepthFFT = FourierTransform((xs, ys), depths)
        
        # collect all the x indices where the depth of the hull is > 0
        # to extract the ship length
        i_indices = [i[0] for i in np.argwhere(depths > 0)]
        xd = xs[i_indices]
        xmin, xmax = np.min(xd), np.max(xd)
        elShip = xmax - xmin
        
        u =  np.sqrt(self.g * elShip) * froude
        kg = self.g/u**2
        
        print(f'u = {u:.2f} k_g = {kg:.2g} lambda_g = {2*np.pi/kg:.2f}')
        
        def integrand(k):
            kx = np.sqrt(kg * k)
            ky = np.sqrt(k**2 - kx**2)
            kvec = np.array((kx, ky))
            coeff = 1j * kx  / (4 * np.pi)
            return coeff \
                * np.exp( 1j * (kx*x + ky*y) ) \
                * shipDepthFFT(kvec) / np.sqrt(1.0 - kg/k)
            
        def realIntegrand(k):
            return integrand(k).real
        
        # limit the max integration bound to be 1000*kg
        return quad_vec(realIntegrand, kg + eps, 1000*kg)[0]
    
    def createVtkPipeline(self, xx, yy, zz):
        
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
            x = i/(ncolors - 1.)
            r = x**2
            g = np.sin(0.5*np.pi*x)**2
            b = min(1, 2*x)
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

        pipeline = {
            'data': data,
            'coords': coords,
            'pts': pts,
            'grid': grid,
            'dataMapper': dataMapper,
            'dataActor': dataActor,
            'lut': lut,
            'cbar': cbar,
            'xyz': xyz,
        }
        
        return pipeline

    
    def show(self, pipeline):

        # show
        ren = vtk.vtkRenderer()
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        # add the actors to the renderer, set the background and size
        ren.AddActor(pipeline['dataActor'])
        ren.AddActor(pipeline['cbar'])
        ren.SetBackground(0.5, 0.5, 0.5)
        renWin.SetSize(900, 600)
        iren.Initialize()
        renWin.Render()
        
        pipeline['ren'] = ren
        pipeline['renWin'] = renWin
        pipeline['iren'] = iren
        
        return pipeline
        
    def savePNG(self, pipeline, filename):
        
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)

        imgFilter = vtk.vtkWindowToImageFilter()
        imgFilter.SetInput(pipeline['renWin'])
        imgFilter.SetInputBufferTypeToRGB()
        imgFilter.Update()

        # writer.SetFileName(filename)
        writer.SetInputConnection(imgFilter.GetOutputPort())
        writer.Write()
        
        pipeline['writer'] = writer
        pipeline['imgFilter'] = imgFilter
        
        return pipeline
        


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
    parser.add_argument('-l', '--shiplength', type=float, default=4.0, help='Length of ship')
    parser.add_argument('-w', '--shipwidth', type=float, default=4.0, help='Width of ship')
    parser.add_argument('-d', '--shipdepth', type=float, default=1.0, help='Depth of ship hull')
    parser.add_argument('-n', '--nx', type=int, default=128, help='Number of cells in x direction')
    parser.add_argument('-m', '--my', type=int, default=64, help='Number of cells in y direction')
    parser.add_argument('-show', '--show', action='store_true', default=False, help='Show the wake field')
    parser.add_argument('-save', '--save', type=str, default='', help='Save the wake field in file')

    args = parser.parse_args()

    print(f'Fr={args.froude} ship len: {args.shiplength}')
    
    sw = ShipWake(elx=args.xlen, nx=args.nx, ny=args.my)
    
    buffer = 0.1 * args.shiplength
    x = np.linspace(-args.xlen + buffer, buffer, args.nx + 1)
    y = np.linspace(0.0, args.xlen/2.0, args.my + 1)
    xx, yy = np.meshgrid(x, y)
    
    # ship disturbance
    elShipL = args.shiplength
    elShipW = args.shipwidth
    # mid location 
    x0, y0 = 0.0, 0.0
    # fixed resolution for the time being
    xs = np.linspace(x0 - 10*elShipL, x0 + 10*elShipL, 128)
    ys = np.linspace(y0 - 10*elShipW, y0 + 10*elShipW, 128)
    xxs, yys = np.meshgrid(xs, ys)
    valid_region = (xxs/elShipL)**2 + (yys/elShipW)**2 < 1.0
    depths = valid_region * args.shipdepth
    
    zzr = sw.compute(xx, yy, xs, ys, depths, froude=args.froude, )
    
    print(f'mean, std height: {zzr.mean()}, {zzr.std()}')

    pipeline = sw.createVtkPipeline(xx, yy, zzr)
    if args.show:
        pipeline = sw.show(pipeline)
        pipeline['iren'].Start()

        
    if args.save:
        # need to add the renders to the pipeline
        pipeline = sw.show(pipeline)
        pipeline = sw.savePNG(pipeline, args.save)

if __name__ == '__main__':
    main()