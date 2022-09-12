#!/usr/bin/env python

# This file is part of ONEBra-core algorithm
# An open source algorithm to generate a 3D model of the bra they need to have a perfectly symmetric breast, starting from a simple 3D photo of the complete breast.
# 
# ONEBra-core is free software: you can redistribute it and/or modify it under the terms of the
# GNU Affiero General Public License version 3 as published by the Free Software Foundation.
# ONEBra-core is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affiero General Public License version 3 for more details.
# You should have received a copy of the Affiero General Public License version 3 along with ONEBra-core.
# If not, see <https://www.gnu.org/licenses/>.

import os.path
import yaml,math,sys

import vtk
from vtk import (
    vtkMassProperties,
    vtkTriangleFilter,
    vtkReflectionFilter,
    vtkDataSetSurfaceFilter,
    vtkIntersectionPolyDataFilter
)
import vtkmodules.all
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonDataModel import (
    vtkPlane,
    vtkPlaneCollection,
    vtkBox,
    vtkSphere,
    vtkPolyData
)
from vtkmodules.vtkCommonCore import (
    VTK_DOUBLE_MAX,
    vtkPoints
)
from vtkmodules.vtkCommonTransforms import (
    vtkTransform
)
from vtkmodules.vtkFiltersGeneral import (
    vtkTransformPolyDataFilter
)
from vtkmodules.vtkFiltersCore import (
    vtkAppendPolyData,
    vtkClipPolyData,
    vtkPolyDataConnectivityFilter
)
from vtkmodules.vtkFiltersSources import vtkCubeSource,vtkCylinderSource,vtkSphereSource
from vtkmodules.vtkIOGeometry import (
    vtkSTLReader,
    vtkSTLWriter
)
import numpy as np

def clipWithSphere(breast, center, radius):
    # clip the input breast with a sphere of given center and radius
    breastImpContext = vtkSphere()
    breastImpContext.SetCenter(center)
    breastImpContext.SetRadius(radius)
    
    refclipper = vtkClipPolyData()
    refclipper.SetInputData(breast)
    refclipper.SetClipFunction(breastImpContext)
    refclipper.InsideOutOn()
    refclipper.Update()
    return refclipper.GetOutput()
    
def clipWithBBox(bounds, polyData, x_left_width, x_right_width, y_up_width, y_down_width, neg_z_tol):
    # clip a polyData using a cube
    xMin, xMax, yMin, yMax, zMin, zMax = bounds
    
    breastBBox = vtkBox()
    # cap a bit more in y and z direction (negative)
    breastBBox.SetBounds( xMin - x_right_width, xMax + x_left_width, yMin - y_down_width, yMax + y_up_width, zMin - neg_z_tol, zMax )
    
    clipper = vtkClipPolyData()
    clipper.SetInputData(polyData)
    clipper.SetClipFunction(breastBBox)
    clipper.InsideOutOn()
    clipper.Update()

    return clipper.GetOutput()
    
def ReadPolyData(file_name):
    import os
    path, extension = os.path.splitext(file_name)
    extension = extension.lower()
    if extension == '.stl':
        reader = vtkSTLReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    else:
        # Return a None if the extension is unknown.
        poly_data = None
    return poly_data

def largestConnectedComponent(polyData):
    # work only on largest connected component
    filt = vtkPolyDataConnectivityFilter()
    filt.SetInputData(polyData)
    filt.SetExtractionModeToLargestRegion()
    filt.Update()

    connectedData = filt.GetOutput()
    return connectedData
    
def blindClip(config, connectedData):    
    # blindly cut the 3D photo with a given parallelepiped
    polyCenter = connectedData.GetCenter()
    bbbox=config['blind_breasts_bounding_box']
    xwidth = bbbox['xwidth']
    ywidth = bbbox['ywidth']
    zwidth = bbbox['zwidth']
    breastsData = clipWithBBox([polyCenter[0]-xwidth, polyCenter[0]+xwidth,
                                polyCenter[1]-ywidth, polyCenter[1]+ywidth,
                                polyCenter[2]-zwidth, polyCenter[2]+zwidth],
                               connectedData, 0, 0, 0, 0, 0)
    return polyCenter, breastsData

def breastsClip(config, polyCenter, breastsData):
    # algorithm to identify the left and right breast automatically
    polyBoundsLeft = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    polyBoundsRight = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    polyNCellsLeft = 0
    polyNCellsRight = 0
    hdiff = 0
    vdiff = 0
    niterations_zero_cells = 0
    max_iterations_zero_cells = 5
    increment = config['increment']
    min_region_size = config['min_region_size']
    while (polyBoundsLeft == polyBoundsRight
           or polyNCellsLeft < min_region_size
           or polyNCellsRight < min_region_size
           or vdiff >= hdiff) and niterations_zero_cells < max_iterations_zero_cells:
        
        plane = vtkPlane()
        # 0.1 no 0.11 si
        plane.SetOrigin(polyCenter[0], polyCenter[1], polyCenter[2])
        plane.SetNormal(0.0, 0.0, 1.0)

        clipper = vtkClipPolyData()
        clipper.SetInputData(breastsData)
        clipper.SetClipFunction(plane)
        clipper.SetValue(0)
        clipper.Update()

        filt = vtkPolyDataConnectivityFilter()
        filt.SetInputData(clipper.GetOutput())
        filt.SetExtractionModeToClosestPointRegion()
        # left breast
        filt.SetClosestPoint(polyCenter[0]-1, polyCenter[1], polyCenter[2])
        filt.Update()
    
        polyDataLeft = filt.GetOutput()
        # get bounding box of left breast
        polyBoundsLeft = polyDataLeft.GetBounds()
        polyCellsLeft = polyDataLeft.GetPolys()
        polyNCellsLeft = polyCellsLeft.GetNumberOfCells()
        polyCenterLeft = polyDataLeft.GetCenter()
        
        filt = vtkPolyDataConnectivityFilter()
        filt.SetInputData(clipper.GetOutput())
        filt.SetExtractionModeToClosestPointRegion()
        # right breast
        filt.SetClosestPoint(polyCenter[0]+1, polyCenter[1], polyCenter[2])
        filt.Update()

        polyDataRight = filt.GetOutput()
        # get bounding box of left breast
        polyBoundsRight = polyDataRight.GetBounds()
        polyCellsRight = polyDataRight.GetPolys()
        polyNCellsRight = polyCellsRight.GetNumberOfCells()
        polyCenterRight = polyDataRight.GetCenter()

        if polyNCellsRight <= 0 and polyNCellsLeft <= 0:
            niterations_zero_cells += 1
        
        hdiff = abs(polyCenterLeft[0] - polyCenterRight[0])
        vdiff = abs(polyCenterLeft[1] - polyCenterRight[1])
        
        # increase z axis cut and retry
        polyCenter = (polyCenter[0], polyCenter[1], polyCenter[2]+increment)
    # decide wich one is the impaired breast

    if niterations_zero_cells >= max_iterations_zero_cells:
        raise Exception("ERROR: Failed to identify left and right breasts, try narrowing blind_breasts_bounding_box widths in the configuration file, and enabling debug options.")
    
    #transform mesh to triangles
    triangleFilterLeft = vtkTriangleFilter()
    triangleFilterLeft.SetInputData(polyDataLeft)
    triangleFilterLeft.Update()
    breastLeft = triangleFilterLeft.GetOutput()

    triangleFilterRight = vtkTriangleFilter()
    triangleFilterRight.SetInputData(polyDataRight)
    triangleFilterRight.Update()
    breastRight = triangleFilterRight.GetOutput()

    surfaceLeft = vtkMassProperties()
    surfaceLeft.SetInputData(breastLeft)
    surfaceLeft.Update()
    surfaceRight = vtkMassProperties()
    surfaceRight.SetInputData(breastRight)
    surfaceRight.Update()
    
    if surfaceLeft.GetSurfaceArea() < surfaceRight.GetSurfaceArea():
        # left one is the impaired one, rotate the right one
        breastRef = breastRight
        breastImp = breastLeft
    else:
        breastRef = breastLeft
        breastImp = breastRight

    return (breastRef, breastImp)

def augmentBreastSurface(config, connectedData, breastRef, breastImp):
    # now that we identified impaired and reference breast
    # start to work with the full  breast (undo cut on z axis)
    bbox= config['breast_bounding_box']
    rxMin, rxMax, ryMin, ryMax, rzMin, rzMax = breastRef.GetBounds()
    rx, ry, rz = breastRef.GetCenter()
    ix, iy, iz = breastImp.GetCenter()
    rxInc,ryInc,rzInc = ((rxMax - rxMin) / 2, (ryMax - ryMin) / 2, (rzMax - rzMin) / 2)
    ixMin, ixMax, iyMin, iyMax, izMin, izMax = (ix-rxInc, ix+rxInc, iy-ryInc, iy+ryInc, iz-rzInc, iz+rzInc)
    breastRefComplete = clipWithBBox([rxMin, rxMax, ryMin, ryMax, rzMin, rzMax], connectedData,
                                     bbox['x_left_width'], bbox['x_right_width'],
                                     bbox['y_up_width'], bbox['y_down_width'], bbox['z_neg_width'])
    breastImpComplete = clipWithBBox([ixMin, ixMax, iyMin, iyMax, izMin, izMax], connectedData, 
                                     bbox['x_left_width'], bbox['x_right_width'],
                                     bbox['y_up_width'], bbox['y_down_width'], bbox['z_neg_width'])

    return (breastRefComplete, breastImpComplete)
    
def reflectBreast(config, breastRefComplete):        
    # reflect reference breast, needed before superimposing
    tpd = vtkReflectionFilter()
    tpd.CopyInputOff()
    tpd.SetInputData(breastRefComplete)
    tpd.Update()

    toPolyData = vtkDataSetSurfaceFilter()
    toPolyData.SetInputData(tpd.GetOutput())
    toPolyData.Update()
    breastRefReflect = toPolyData.GetOutput()
    return breastRefReflect

def alignBreasts(config, breastRefComplete, breastImpComplete, breastRefReflect):    
    # superimpmose impaired and reference breasts
    breastImpBbox=breastImpComplete.GetBounds() 
    breastRefBbox=breastRefComplete.GetBounds()
    breastRefReflectBbox = breastRefReflect.GetBounds()
    ImpWidth = breastImpBbox[1] - breastImpBbox[0]
    RefWidth = breastRefBbox[1] - breastRefBbox[0]
    
    xTransform = vtkTransform()
    xTransform.Translate(breastImpBbox[1] - breastRefReflectBbox[1]+ abs(RefWidth-ImpWidth) + config['breasts_align_x_offset'], config['breasts_align_y_offset'], config['breasts_align_z_offset'])
    
    transform = vtkTransformPolyDataFilter()
    transform.SetInputData(breastRefReflect)
    transform.SetTransform(xTransform)
    transform.Update()
    breastRefReflectAlign = transform.GetOutput()
    return breastRefReflectAlign

def sphereClip(config, breastRefComplete, breastImpComplete, breastRefReflectAlign, breastImpAlign):
    # clip the final cup with a sphere
    breastImpBbox=breastImpComplete.GetBounds() 
    breastRefBbox=breastRefComplete.GetBounds()
    rxMin, rxMax, ryMin, ryMax, rzMin, rzMax = breastRefBbox
    # radius of equivalent circle in xy, xz and yz subplanes
    rRadius = ( math.sqrt((rxMax - rxMin)**2 + (ryMax - ryMin)**2) / 2, math.sqrt((rxMax - rxMin)**2 + (rzMax - rzMin)**2) / 2, math.sqrt((rzMax - rzMin)**2 + (ryMax - ryMin)**2) / 2)
    ImpWidth = breastImpBbox[1] - breastImpBbox[0]
    RefWidth = breastRefBbox[1] - breastRefBbox[0]
    rx, ry, rz = breastRefReflectAlign.GetCenter()
    ix, iy, iz = breastImpAlign.GetCenter()
    center = [(rx+ix)/2, (ry+iy)/2, (rz+iz)/2]
    radius = rRadius[0]*config['breast_clip']
    breastImpSplit = clipWithSphere(breastImpAlign, center, radius)
    breastRefSplit = clipWithSphere(breastRefReflectAlign, center, radius)

    return (breastImpSplit, breastRefSplit)

def writeSTL(output_file, polyDatas):
    appendAll = vtkAppendPolyData()
    for polyData in polyDatas:
        appendAll.AddInputData(polyData)

    writer = vtkSTLWriter()
    writer.SetFileName(output_file)
    writer.SetInputConnection(appendAll.GetOutputPort())
    writer.Write()

def main():
    cpath = None
    if len(sys.argv) <= 1:
        # defaul look in curdir for onebra.yml
        cpath = os.path.join( os.curdir, 'onebra.yaml' )
    else:
        cpath = sys.argv[1]
        
    with open(cpath, 'r') as f:
        config = yaml.safe_load(f)
    
    # Define colors
    colors = vtkNamedColors()
    backgroundColor = colors.GetColor3d('steel_blue')
    impColor = colors.GetColor3d('Banana')
    refColor = colors.GetColor3d('Tomato')

    if config['input_stl_file_path'] and os.path.isfile(config['input_stl_file_path']):
        polyData = ReadPolyData(config['input_stl_file_path'])
    else:
        print("Invalid input file: ")
        print(config['input_stl_file_path'])
        sys.exit(1)
    
    increment = config['increment']

    connectedData = largestConnectedComponent(polyData)
    breastsDataCenter, breastsData = blindClip(config, connectedData)
                
    try:
        breastRef, breastImp = breastsClip(config, breastsDataCenter, breastsData)
    except Exception as e:
        print(str(e))
        sys.exit(1)
        
    breastRefComplete,breastImpComplete = augmentBreastSurface(config, connectedData, breastRef, breastImp)
    breastRefReflect = reflectBreast(config, breastRefComplete)
    
    # superimpose the two breasts
    breastRefReflectAlign = alignBreasts(config, breastRefComplete, breastImpComplete, breastRefReflect)

    # clip again to get rid of eventual space left after breasts_align_x/y/z_offset
    breastImpAlign = clipWithBBox(breastRefReflectAlign.GetBounds(), breastImpComplete,
                                  0, 0, 0, 0, 0.1)
    
    # clip on a sphere around impaired breast and superimpose reference breast
    breastImpSplit, breastRefSplit = sphereClip(config, breastRefComplete, breastImpComplete, breastRefReflectAlign, breastImpAlign)

    # write bra
    writeSTL(config['output_stl_bra_file_path'], [breastImpSplit, breastRefSplit])
    
if __name__ == '__main__':
    main()
