import sys, shutil, math, pdb
import os as os
import numpy as np
from mpi4py import MPI as pyMPI
from dolfin import *
import dolfin as dolfin
from fenicstools import *

import vtk_py3
import vtk


from ..utils.oops_objects_MRC2 import State_Variables
from ..utils.oops_objects_MRC2 import load_ischemia_mask
from ..ep.EPmodel_basic_test import EPmodel
# from ..ep.EPmodel_cpp import EPmodel



def createEPmodel(IODet, SimDet):

    directory_ep = IODet["directory_ep"]
    outputfolder = IODet["outputfolder"]
    folderName = IODet["folderName"] + IODet["caseID"] + "/"
    delTat = SimDet["dt"]

    if "casename_ep" in IODet:
        casename = IODet["casename_ep"]
    else:
        casename = IODet["casename"]

    casename_marked = IODet["casename_ep_marked"]          

    if "isFCH" in list(SimDet.keys()):
        isFCH = SimDet["isFCH"]
    else:
        isFCH = False
    if "iswaorta" in list(SimDet.keys()):
        iswaorta = SimDet["iswaorta"]
    else:
        iswaorta = False  # Default

    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    # Read EP data from HDF5 Files
    mesh_ep = Mesh()
    comm_common = mesh_ep.mpi_comm()

    meshfilename_ep = directory_ep + casename + ".hdf5"
    f = HDF5File(comm_common, meshfilename_ep, "r")
    f.read(mesh_ep, casename, False)
    if isFCH:
        mesh_ep.scale(6.5e-2)
    elif iswaorta:
        mesh_ep.scale(1.1)
        # pass

    File(outputfolder + folderName + "mesh_ep.pvd") << mesh_ep

    facetboundaries_ep = MeshFunction("size_t", mesh_ep, 2)
    f.read(facetboundaries_ep, casename + "/" + "facetboundaries")

    matid_ep = MeshFunction("size_t", mesh_ep, mesh_ep.topology().dim())
    AHAid_ep = MeshFunction("size_t", mesh_ep, mesh_ep.topology().dim())

    if f.has_dataset(casename + "/" + "matid"):
        if SimDet.get("function_matid"):
            VQuadelem = FiniteElement("DG", mesh_ep.ufl_cell(), degree=0, quad_scheme="default")
            matid_FS = FunctionSpace(mesh_ep, VQuadelem)
            matid_func = dolfin.Function(matid_FS)
            for cell in cells(mesh_ep):
                matid_ep[cell.index()] = round(matid_func(cell.midpoint()))
        else:
            f.read(matid_ep, casename + "/" + "matid")
    else:
        matid_ep.set_all(0)

    if f.has_dataset(casename + "/" + "AHAid"):
        f.read(AHAid_ep, casename + "/" + "AHAid")
    else:
        AHAid_ep.set_all(0)

    mesh_ep_marked = Mesh()
    comm_common_marked = mesh_ep_marked.mpi_comm()
    meshfilename_ep_marked = directory_ep + casename_marked + ".hdf5"
    f_marked = HDF5File(comm_common_marked, meshfilename_ep_marked, "r")
    f_marked.read(mesh_ep_marked, casename_marked, False)

    # Load ischemia mask for EP mesh
    ischemia_mask_ep = load_ischemia_mask(mesh_ep_marked, directory_ep, casename_marked)

    deg_ep = 4

    Quadelem_ep = FiniteElement("Quadrature", mesh_ep.ufl_cell(), degree=deg_ep, quad_scheme="default"    )
    Quadelem_ep._quad_scheme = "default"
    Quad_ep = FunctionSpace(mesh_ep, Quadelem_ep)

    if "fiber_fspace" in list(SimDet.keys()) and "fiber_fspace_deg" in list(
        SimDet.keys()):
        VQuadelem_ep = VectorElement(
            SimDet["fiber_fspace"],
            mesh_ep.ufl_cell(),
            degree=SimDet["fiber_fspace_deg"],
            quad_scheme="default",
        )
        VQuadelem_ep._quad_scheme = "default"
    else:
        VQuadelem_ep = VectorElement(
            "Quadrature", mesh_ep.ufl_cell(), degree=deg_ep, quad_scheme="default"
        )
        VQuadelem_ep._quad_scheme = "default"

    fiberFS_ep = FunctionSpace(mesh_ep, VQuadelem_ep)

    f0_ep = Function(fiberFS_ep)
    s0_ep = Function(fiberFS_ep)
    n0_ep = Function(fiberFS_ep)

    if SimDet["DTI_EP"] is True:
        f.read(f0_ep, casename + "/" + "eF_DTI")
        f.read(s0_ep, casename + "/" + "eS_DTI")
        f.read(n0_ep, casename + "/" + "eN_DTI")
    else:
        f.read(f0_ep, casename + "/" + "eF")
        f.read(s0_ep, casename + "/" + "eS")
        f.read(n0_ep, casename + "/" + "eN")

    f.close()

    # # --------------------------------------------------------
    # # LOAD ISCHEMIA MASK from the *_marked.hdf5 file
    # # --------------------------------------------------------
    # ischemia_mask_ep = MeshFunction("size_t", mesh_ep, mesh_ep.topology().dim(), 0)

    # if casename_marked is not None:
    #     maskfile = os.path.join(directory_ep, casename_marked + ".hdf5")
    #     if os.path.exists(maskfile):
    #         try:
    #             fm = HDF5File(comm_common, maskfile, "r")
    #             fm.read(ischemia_mask_ep, casename_marked + "/ischemia_mask")
    #             fm.close()
    #             print(f"Loaded EP ischemia_mask from {maskfile}")
    #         except Exception as e:
    #             print("⚠️ Failed to read EP ischemia_mask, using all-zero mask.", e)
    #     else:
    #         print(f"⚠️ No marked EP file {maskfile}, using all-zero mask.")    


    comm_ep = mesh_ep.mpi_comm()

    # Define state variables
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    state_obj = State_Variables(comm_ep, SimDet)
    state_obj.dt.dt = delTat
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

    if "d_iso" in list(SimDet.keys()):
        d_iso = SimDet["d_iso"]
    else:
        d_iso = 0.02

    if "d_ani_factor" in list(SimDet.keys()):
        d_ani_factor = SimDet["d_ani_factor"]
    else:
        d_ani_factor = 0.02

    if "ani_factor" in list(SimDet.keys()):
        ani_factor = SimDet["ani_factor"]
    else:
        ani_factor = 1000.0

    if "CRT" in list(SimDet.keys()):
        CRT = SimDet["CRT"]
    else:
        CRT = False

    if "CRT_current_intensity" in list(SimDet.keys()):
        CRT_current_intensity = SimDet["CRT_current_intensity"]
    else:
        CRT_current_intensity = 0.0        

    if "CRT_pos" in list(SimDet.keys()):
        CRT_pos = SimDet["CRT_pos"]
    else:
        CRT_pos = [0.0,0.0,0.0]

    if "CRT_pacing_timing" in list(SimDet.keys()):
        CRT_pacing_timing = SimDet["CRT_pacing_timing"]
    else:
        CRT_pacing_timing = [[0.0, 0.0]]



    EPparams = {
        "EPmesh": mesh_ep,
        "deg": 4,
        "matid": matid_ep,
        "facetboundaries": facetboundaries_ep,
        "f0": f0_ep,
        "s0": s0_ep,
        "n0": n0_ep,
        "state_obj": state_obj,
        "d_iso": d_iso,
        "d_ani": d_ani_factor,
        "ani_factor": ani_factor,
        # "ploc": SimDet["ploc"],
        "AHAid": AHAid_ep,
        "matid": matid_ep,
        "Ischemia": SimDet["Ischemia"],
        "Ischemia_mask": ischemia_mask_ep,
        "Ischemic_stiffness_factor": SimDet["Ischemic_stiffness_factor"],  
        "Ischemic_ep_factor": SimDet["Ischemic_ep_factor"],     
        "abs_tol": SimDet["abs_tol"],
        "rel_tol": SimDet["rel_tol"],
        "CRT": CRT,  
        "CRT_current_intensity": CRT_current_intensity,
        "CRT_pos": CRT_pos,   
        "CRT_pacing_timing": CRT_pacing_timing,
        "CRT_eps_factor": SimDet["CRT_eps_factor"],   
    }

    if "ploc" in list(SimDet.keys()):
        EPparams.update({"ploc": SimDet["ploc"]})



    if "pacing_timing" in list(SimDet.keys()):
        EPparams.update({"pacing_timing": SimDet["pacing_timing"]})

    # if "isPJ" in list(SimDet.keys()):
    #     if SimDet["isPJ"] and "pj_tnodes" in list(SimDet.keys()):
    #         EPparams.update({"ploc": SimDet["pj_tnodes"]})

    if "isPJ" in SimDet and SimDet["isPJ"] and "pj_tnodes" in SimDet:
        pj_tnodes = SimDet["pj_tnodes"]
    else:
        pj_tnodes = []

    crt_pos    = SimDet.get("CRT_pos", []) if SimDet.get("CRT", False) else []

    # Merge both pacing sets into a single ploc list
    ploc_all = list(pj_tnodes) + list(crt_pos)

    # Record counts for later index separation
    nPJ  = len(pj_tnodes)
    nCRT = len(crt_pos)

    # --- Update EPparams ---
    EPparams.update({
        "ploc": ploc_all,
        "nPJ":  nPJ,
        "nCRT": nCRT
    })    

    if SimDet["isPJ"] and "PJ_current_intensity" in list(SimDet.keys()):
        EPparams.update({"current_intensity": SimDet["current_intensity"]})

    if "Ischemia" in list(SimDet.keys()):
        EPparams.update({"Ischemia": SimDet["Ischemia"]})

    # Attach to params for use by EPModel
    EPparams["Ischemia_mask"] = ischemia_mask_ep    

    # Define EP model and solver
    EPmodel_ = EPmodel(EPparams)

    return EPmodel_, state_obj, EPparams

