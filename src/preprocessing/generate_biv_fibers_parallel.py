#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate BiV fiber/sheet directions in parallel using SetBiVFiber_Quad_PyQ.
"""

import os, sys, argparse, pdb
sys.setrecursionlimit(5000)

sys.path.append("/mnt/Research")
sys.path.append("/mnt/Research/vtk_py3")
sys.path.append("/mnt/Output")
sys.path.append("/mnt/Research/heArt")
sys.path.append("/home/fenicstools")

from mpi4py import MPI
import dolfin as df
import numpy as np

import SetBiVFiber_Quad_PyQ as fibers


# def parse_args():
#     p = argparse.ArgumentParser(description="Parallel BiV fiber generation")
#     p.add_argument("--h5-mesh", required=True)
#     p.add_argument("--casename", required=True)
#     p.add_argument("--outfolder", required=True)
#     p.add_argument("--caseID", required=True)

#     # facet ids
#     p.add_argument("--LVendoid", type=int, default=3)
#     p.add_argument("--RVendoid", type=int, default=1)
#     p.add_argument("--epiid",    type=int, default=2)
#     p.add_argument("--topid",    type=int, default=4)
#     p.add_argument("--LVtopid",  type=int, default=None)
#     p.add_argument("--RVtopid",  type=int, default=None)

#     # matids
#     p.add_argument("--LV-matid", type=int, default=0)
#     p.add_argument("--Septum-matid", type=int, default=1)
#     p.add_argument("--RV-matid", type=int, default=2)

#     p.add_argument("--LV-fiber-angle", nargs=2, type=float, default=[60,-60])
#     p.add_argument("--RV-fiber-angle", nargs=2, type=float, default=[60,-60])
#     p.add_argument("--Septum-fiber-angle", nargs=2, type=float, default=[60,-60])
#     p.add_argument("--LV-sheet-angle", nargs=2, type=float, default=[0.1, -0.1])
#     p.add_argument("--RV-sheet-angle", nargs=2, type=float, default=[0.1, -0.1])
#     p.add_argument("--Septum-sheet-angle", nargs=2, type=float, default=[0.1, -0.1])

#     p.add_argument("--degree", type=int, default=4)
#     p.add_argument("--isrotatept", action="store_true")

#     return p.parse_args()


def generate_biv_fibers_parallel(IODet, SimDet):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ----------------------------------------------------------
    # INPUTS FROM IODet
    # ----------------------------------------------------------
    directory_me = IODet["directory_me"]
    casename_me  = IODet["casename_me"]
    caseID       = IODet.get("fiber_caseID", IODet.get("caseID"))    

    # Mesh path
    h5_mesh = os.path.join(directory_me, casename_me + ".hdf5")

    if rank == 0:
        print("===== BiV Fiber Generation =====")
        print("Mesh file:", h5_mesh)
        print("CaseID:", caseID)

    # ---------------------- READ MESH ---------------------------
    mesh = df.Mesh(comm)
    h5 = df.HDF5File(comm, h5_mesh, "r")    

    mesh_path  = f"{casename_me}"
    facet_path = f"{casename_me}/facetboundaries"
    matid_path = f"{casename_me}/matid"

    h5.read(mesh, mesh_path, False)

    facetboundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    h5.read(facetboundaries, facet_path)

    matid = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    h5.read(matid, matid_path)

    h5.close()

    if rank == 0:
        print("Cells:", mesh.num_cells())
        try:
            print("Matid unique:", np.unique(matid.array()))
        except:
            pass


    if rank == 0:
        print("Cells:", mesh.num_cells())
        print("Matid unique:", np.unique(matid.array()))


    # ----------------------------------------------------------
    # PARAMETERS FROM SimDet
    # ----------------------------------------------------------
    LVendoid = SimDet.get("LVendoid")
    RVendoid = SimDet.get("RVendoid")
    epiid    = SimDet.get("epiid")
    topid    = SimDet.get("topid")
    LVtopid  = SimDet.get("LVtopid")
    RVtopid  = SimDet.get("RVtopid")



    # ---------------------- PARAMETERS FOR FIBERS ------------------------
    baseid = [topid]
    if LVtopid is not None: baseid.append(LVtopid)
    if RVtopid is not None: baseid.append(RVtopid)

    outdir = os.path.join(directory_me)

    if rank == 0:
        os.makedirs(outdir, exist_ok=True)

    comm.Barrier()

    # ----------------------------------------------------------
    # PARAMETERS FOR FIBER GENERATION
    # (using exactly your hard-coded values)
    # ----------------------------------------------------------   

    params ={"mesh": mesh,\
                    "facetboundaries": facetboundaries,\
                    "LV_fiber_angle": [0.01,-0.01], \
                    "LV_sheet_angle": [0.1, -0.1], \
                    "Septum_fiber_angle": [0.01, -0.01],\
                    "Septum_sheet_angle": [0.1, -0.1],\
                    "RV_fiber_angle": [0.01, -0.01],\
                    "RV_sheet_angle": [0.1, -0.1],\
                    "LV_matid": 0,\
                    "Septum_matid": 1,\
                    "RV_matid": 2,\
                    "matid":  matid,\
                    "isrotatept": False,\
                    "isreturn": True,\
                    # "outfilename":  casename,
                    "outdirectory": outdir,
                    "baseid": baseid,
                    "epiid": epiid,
                    "rvid": RVendoid,
                    "lvid": LVendoid,
                    "degree": 4}

    # ---------------------- CALL FIBER GENERATOR ------------------------
    if rank == 0:
        print("Computing fibers...")

    eC0, eL0, eR0 = fibers.SetBiVFiber_Quad_PyQ(params)

    # ---------------------- SAVE WITH MATCHING PATH ---------------------
    out_h5 = os.path.join(outdir, "BiV_fibers.h5")

    if rank == 0:
        print("Saving CG1 fibers to:", out_h5)

    h5f = df.HDF5File(comm, out_h5, "w")
    h5f.write(mesh, f"BiV_fibers")
    h5f.write(eC0, f"BiV_fibers/eC0")
    h5f.write(eL0, f"BiV_fibers/eL0")
    h5f.write(eR0, f"BiV_fibers/eR0")
    h5f.close()

    if rank == 0:
        print("Done.")