import vtk
import numpy

def show(xx, yy, zz) -> None:
    
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
        x = 0.5*i*numpy.pi/(ncolors - 1.)
        r = numpy.cos(3*x)**2
        g = numpy.cos(1*x)**2
        b = numpy.cos(5*x)**2
        a = 1.0 # opacity
        lut.SetTableValue(i, r, g, b, a)
    lut.SetTableRange(fMin, fMax)
    cbar.SetLookupTable(lut)
    dataMapper.SetUseLookupTableScalarRange(1)

    # construct the grid
    grid.SetDimensions(nx1, ny1, 1)

    xyz = numpy.empty((numPoints, 3), float)
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

    x = numpy.linspace(0., 1., 11)
    y = numpy.linspace(-1, 2., 21)
    xx, yy = numpy.meshgrid(x, y)
    zz = xx
    show(xx, yy, zz)