import sys
import vtk
sys.path.append("/mnt/Research")
import vtk_py3 as vtk_py
from dolfin import *
import pdb
import numpy as np

def markPJ_cells(IODet, SimDet):

    marked_cells_array = [SimDet["lbbb_location"]] #[7,8, 11, 13, 14]#[]#6, 7, 8, 9, 10, 11, 12, 15, 16]

    mesh_pj = Mesh()
    comm_pj = mesh_pj.mpi_comm()

    casename = IODet["casename_pj"] #"PJ"
    f = HDF5File(comm_pj, IODet["directory_pj"]+ "/" + casename+".hdf5", "r")
    f.read(mesh_pj, casename, False)

    cell_markers = MeshFunction("size_t", mesh_pj, mesh_pj.topology().dim(), 0)
    for cell in cells(mesh_pj):
        if(cell.index() in marked_cells_array):
            print("marked")
            cell_markers[cell] = 1  # Only mark relevant local cells

    File(IODet["directory_pj"]+ "/" + "markedcells.pvd") << cell_markers

    hdf5_file =  "PJmarked"  + ".hdf5"
    case_name = "PJmarked" 
    f = HDF5File(MPI.comm_world, IODet["directory_pj"]+ "/" + hdf5_file, 'w')
    f.write(mesh_pj, case_name)
    f.close()

    f = HDF5File(MPI.comm_world, IODet["directory_pj"]+ "/" + hdf5_file, 'a')
    f.write(cell_markers, case_name+'/'+'matid')
    f.close()

    return
 
