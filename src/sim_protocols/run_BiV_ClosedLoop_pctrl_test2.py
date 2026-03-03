import sys, shutil, math, pdb
import os as os
import numpy as np
from mpi4py import MPI as pyMPI
from .coupleEPandPJ import coupleEPandPJ
from .coupleEPandCRT import coupleEPandCRT
from .createEPmodel import createEPmodel
from .createPJmodel import createPJmodel
from scipy.spatial import cKDTree

import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)


from dolfin import *
import dolfin
from fenicstools import *

import vtk_py3
import vtk

from ..utils.oops_objects_MRC2 import printout
from ..utils.oops_objects_MRC2 import biventricle_mesh as biv_mechanics_mesh
from ..utils.oops_objects_MRC2 import lv_mesh as lv_mechanics_mesh

from ..utils.oops_objects_MRC2 import State_Variables
from ..utils.oops_objects_MRC2 import update_mesh
from ..utils.oops_objects_MRC2 import exportfiles
from ..utils.oops_objects_MRC2 import load_ischemia_mask
from ..utils.oops_objects_MRC2 import build_active_indicator

from ..utils.mesh_scale_create_fiberFiles import create_EDFibers
from ..utils.oops_objects_MRC2 import json_serialize

from ..ep.EPmodel_basic_test import EPmodel
#from ..ep.EPmodel_cpp import EPmodel

from ..mechanics.MEmodel3 import MEmodel

# from ..mechanics.MEmodel_pctrl import MEmodel
from .circ import CLmodel
from .circBiV import CLmodel as CLmodel_biv

# from ..mechanics.volume_ca import MeshModifier
from ..postprocessing.postprocessdatalib2 import *

import json

from ..mechanics.JRp import *


def run_BiV_ClosedLoop(IODet, SimDet):
    if "fiber_fspace_deg" in SimDet:
        deg = SimDet["fiber_fspace_deg"]
    else:
        deg = 4
    flags = ["-O3", "-ffast-math", "-march=native"]
    parameters["form_compiler"]["representation"] = "uflacs"
    parameters["form_compiler"]["quadrature_degree"] = deg
    casename_me = IODet["casename_me"]
    casename_me_marked = IODet["casename_me_marked"]    
    casename_ep = IODet["casename_ep"]
    casename_ep_marked = IODet["casename_ep_marked"]    
    directory_me = IODet["directory_me"]
    outputfolder = IODet["outputfolder"]
    folderName = IODet["folderName"] + IODet["caseID"] + "/"

    if "HomogenousActivation" in list(SimDet["GiccioneParams"].keys()):
        ishomo = SimDet["GiccioneParams"]["HomogenousActivation"]
    else:
        ishomo = True

    if "isLV" in list(SimDet.keys()):
        isLV = SimDet["isLV"]
    else:
        isLV = False  # Default
    if "iswaorta" in list(SimDet.keys()):
        iswaorta = SimDet["iswaorta"]
    else:
        iswaorta = False  # Default
    if "isFCH" in list(SimDet.keys()):
        isFCH = SimDet["isFCH"]
    else:
        isFCH = False  # Default
    if "isBiV" in list(SimDet.keys()):
        isBiV = SimDet["isBiV"]
    else:
        isBiV = False  # Default
    if "isPJ" in list(SimDet.keys()):
        isPJ = SimDet["isPJ"]
    else:
        isPJ = False

    delTat = SimDet["dt"]

    if not ishomo:
        intensity = SimDet["current_intensity"]

    if isPJ:
        pj_intensity = SimDet["PJ_current_intensity"]

    # Define Purkinje terminal nodes
    if isPJ:
        if "pj_tnodes" in list(SimDet.keys()):
            if isinstance(SimDet["pj_tnodes"], str):
                pj_t_nodes = np.genfromtxt(os.path.join(IODet["directory_pj"], SimDet["pj_tnodes"]), delimiter=',')
            elif isinstance(SimDet["pj_tnodes"], list):
                pj_t_nodes = np.array(SimDet["pj_tnodes"])

            SimDet.update({"pj_tnodes": list(pj_t_nodes)})

        # Create Purkinje PJ model
        EPmodel_pj, state_obj_pj, PJparams = createPJmodel(IODet, SimDet)
        solver_FHN_pj = EPmodel_pj.Solver()

    # Create Tissue EP model
    EPmodel_ep, state_obj, EPparams = createEPmodel(IODet, SimDet)
    solver_FHN_ep = EPmodel_ep.Solver()
    comm_common = EPmodel_ep.mesh.mpi_comm()
    comm_ep = EPmodel_ep.mesh.mpi_comm()

    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    # Mechanics Mesh
    mesh_me = Mesh()
    mesh_me_params = {
        "directory": directory_me,
        "casename": casename_me,
        "casename_marked": casename_me_marked,        
        "fibre_quad_degree": 4,
        "outputfolder": outputfolder,
        "foldername": folderName,
        "state_obj": state_obj,
        "common_communicator": comm_common,
        "MEmesh": mesh_me,
        "isLV": isLV,
    }

    MEmodel_ = MEmodel(mesh_me_params, SimDet)
    solver_elas = MEmodel_.Solver()
    comm_me = MEmodel_.mesh_me.mpi_comm()

    # Define export for exporting data
    export = exportfiles(comm_me, comm_ep, IODet, SimDet)
    # export.exportVTKobj("facetboundaries_me.pvd", MEmodel_.facetboundaries_me)

    # F_ED = Function(MEmodel_.TF)

    if "AHA_segments" in list(SimDet.keys()):
        AHA_segments = SimDet["AHA_segments"]
    else:
        AHA_segments = [0]

    # Get Unloaded volumes
    V_LV_unload = MEmodel_.GetLVV()
    V_RV_unload = MEmodel_.GetRVV()

    printout("V_LV_unload = " + str(V_LV_unload), comm_me)
    printout("V_RV_unload = " + str(V_RV_unload), comm_me)

    nloadstep = SimDet["nLoadSteps"]

    # Unloading LV to get new reference geometry
    MEmodel_.LVCavityvol.vol = MEmodel_.GetLVV()
    MEmodel_.LVCavitypres.pres = 0.0
    MEmodel_.RVCavitypres.pres = 0.0
    MEmodel_.AortaCavitypres.pres = 0.0

    # export.writePV(MEmodel_, 0);
    export.hdf.write(MEmodel_.mesh_me, "ME/mesh")
    export.hdf.write(EPmodel_ep.mesh, "EP/mesh")
    if isPJ:
        export.hdf.write(EPmodel_pj.mesh, "PJ/mesh")

    ## Dump input file
    export.dump_input_file()

    default_params = {
        "EDP": 12.0,
        "maxit": 20,
        "restol": 1e-3,
        "drestol": 1e-4,
        "EDPtol": 1e-1,
        "preinc": 1,
        "LVangle": [60, -60],
    }


    if SimDet.get("EDP"):
        EDP = SimDet["EDP"]
    else:
        EDP = default_params["EDP"]

    LVangle = default_params["LVangle"]
    maxit = default_params["maxit"]
    restol = default_params["restol"]
    drestol = default_params["drestol"]
    EDPtol = default_params["EDPtol"]
    preinc = default_params["preinc"]

    it = 0
    #MEmodel_.isspringon = 1.0

    if isPJ:
        coupleEPandPJ_ = coupleEPandPJ(EPmodel_ep, EPmodel_pj, EPparams, PJparams, state_obj)

    if SimDet.get("CRT", False):
        coupleEPandCRT_ = coupleEPandCRT(EPmodel_ep, EPparams, state_obj)

    print(SimDet["epiid_Kadj_coeff"])
    print(SimDet["dashpotparam"])    

    while 1:
        printout("Loading", comm_me)
        if not SimDet.get("fch_lumped") and not SimDet.get("lv_lumped"):
            MEmodel_.LVCavitypres.pres += (EDP / 0.0075) / nloadstep
        if SimDet.get("aorta_pres"):
            MEmodel_.AortaCavitypres.pres += (EDP * 10.0 / 0.0075) / nloadstep

        if isBiV or isFCH:
            if SimDet.get("fch_fe"):
                MEmodel_.RVCavitypres.pres += (EDP / 0.0075) / nloadstep
                MEmodel_.LACavitypres.pres += (EDP / 5.0 / 0.0075) / nloadstep
                MEmodel_.RACavitypres.pres += (EDP / 2.0 / 0.0075) / nloadstep
            elif SimDet.get("fch_lumped"):
                pass
            else:
                if SimDet.get("RVEDPfactor"):
                    MEmodel_.RVCavitypres.pres += (SimDet["RVEDPfactor"] * EDP / 0.0075) / nloadstep
                else:
                    MEmodel_.RVCavitypres.pres += (0.75 * EDP / 0.0075) / nloadstep
        if not SimDet.get("fch_lumped") and not SimDet.get("lv_lumped"):
            solver_elas.solvenonlinear()

        export.writePV(MEmodel_, 0)
        export.hdf.write(MEmodel_.GetDisplacement(), "ME/u_loading", it)
        it += 1


        printout(
            "LV Pressure = "
            + str(MEmodel_.GetLVP() * 0.0075)
            + " LV Vol = "
            + str(MEmodel_.GetLVV())  # GetVolumeComputation()),
            + "RV Pressure = "
            + str(MEmodel_.GetRVP() * 0.0075)
            + "RV Vol = "
            + str(MEmodel_.GetRVV()),  # GetVolumeComputation()),
            comm_me,
        )

        if SimDet.get("fch_lumped") or SimDet.get("lv_lumped"):
            break
        if MEmodel_.LVCavitypres.pres * 0.0075 >= EDP:
            break

    #printout("volume = " + str(MEmodel_.GetLVV()), comm_me)

    # Assign displacement at end of loading (LCL)
    if "springref" in SimDet.keys():
        readh5fieldvar(SimDet["springref"][0], SimDet["springref"][1], MEmodel_.u_me_ED)
    else:
        MEmodel_.u_me_ED.assign(MEmodel_.GetDisplacement())
        #print("nothing")
    MEmodel_.isspringon = 1.0

    printout("volume = " + str(MEmodel_.GetLVV()), comm_me)

    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    # Declare communicator based on mpi4py
    # eCC, eRR, eLL, deformedMesh, deformedBoundary = MEmodel_.GetDeformedBasis({})

    # fStrain = MEmodel_.GetFiberstrain(F_ED)
    fStrain_uL = MEmodel_.GetFiberstrainUL()
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

    # Closed-loop phase
    stop_iter = SimDet["closedloopparam"]["stop_iter"]

    # Systemic circulation

    # Pulmonary circulation
    heart_shape = [isLV, iswaorta, isFCH]
    if all(not x for x in heart_shape):
        # if not isLV and not iswaorta and not isFCH:
        Cpa = SimDet["closedloopparam"]["Cpa"]
        Cpv = SimDet["closedloopparam"]["Cpv"]
        Vpa0 = SimDet["closedloopparam"]["Vpa0"]
        Vpv0 = SimDet["closedloopparam"]["Vpv0"]
        Rpv = SimDet["closedloopparam"]["Rpv"]
        Rtv = SimDet["closedloopparam"]["Rtv"]
        Rpa = SimDet["closedloopparam"]["Rpa"]
        Rpvv = SimDet["closedloopparam"]["Rpvv"]
        V_pv = SimDet["closedloopparam"]["V_pv"]
        V_pa = SimDet["closedloopparam"]["V_pa"]
        V_RA = SimDet["closedloopparam"]["V_RA"]

    isrestart = 0
    prev_cycle = 0
    cnt = 0

    Qtv = 0
    Qpa = 0
    Qpv = 0
    Qpvv = 0
    Qlvad = 0
    Qlara = 0

    if "Q_tv" in list(SimDet["closedloopparam"].keys()):
        Qtv = SimDet["closedloopparam"]["Q_tv"]
    if "Q_pa" in list(SimDet["closedloopparam"].keys()):
        Qpa = SimDet["closedloopparam"]["Q_pa"]
    if "Q_pv" in list(SimDet["closedloopparam"].keys()):
        Qpv = SimDet["closedloopparam"]["Q_pv"]
    if "Q_pvv" in list(SimDet["closedloopparam"].keys()):
        Qpvv = SimDet["closedloopparam"]["Q_pvv"]
    if "Q_lvad" in list(SimDet["closedloopparam"].keys()):
        Qlvad = SimDet["closedloopparam"]["Q_lvad"]
    if "Q_lara" in list(SimDet["closedloopparam"].keys()):
        Qlara = SimDet["closedloopparam"]["Q_lara"]

    # Parameters for LVAD #############################################
    LVADrpm = 0
    LVADscale = 0
    if "Q_lvad_rpm" in list(SimDet["closedloopparam"].keys()):
        LVADrpm = SimDet["closedloopparam"]["Q_lvad_rpm"]
    if "Q_lvad_scale" in SimDet["closedloopparam"].keys():
        LVADscale = SimDet["closedloopparam"]["Q_lvad_scale"]
    if "Q_lvad_characteristic" in list(SimDet["closedloopparam"].keys()):
        QLVADFn = SimDet["closedloopparam"]["Q_lvad_characteristic"]

    Qlad = 0
    Qlcx = 0

    # Parameters for Shunt #############################################
    Shuntscale = 0.0
    Rsh = 1e9
    if "Shunt_scale" in list(SimDet["closedloopparam"].keys()):
        Shuntscale = SimDet["closedloopparam"]["Shunt_scale"]
    if "Rsh" in list(SimDet["closedloopparam"].keys()):
        Rsh = SimDet["closedloopparam"]["Rsh"]

    potential_me = Function(FunctionSpace(MEmodel_.mesh_me, "DG", 0))
    writecnt = 0

    P_LV = MEmodel_.GetLVP()  # LVCavitypres.pres
    V_LV = MEmodel_.GetLVV()  # GetVolumeComputation()

    if isBiV or isFCH:
        if SimDet.get("fch_fe"):
            P_LA = MEmodel_.GetLAP()  # set an initial value for P_LA and P_RA
            V_LA = MEmodel_.GetLAV()
            P_RA = MEmodel_.GetRAP()
            V_RA = MEmodel_.GetRAV()

            P_RV = MEmodel_.GetRVP()
            V_RV = MEmodel_.GetRVV()
        elif SimDet.get("fch_lumped"):
            P_LV = EDP / 0.0075
            P_RV = EDP / 0.0075
        else:
            P_RV = MEmodel_.GetRVP()  # LVCavitypres.pres
            V_RV = MEmodel_.GetRVV()  # GetVolumeComputation()

    if isLV or iswaorta:
        if SimDet.get("lv_lumped"):
            CLmodel_ = CLmodel(SimDet)
        else:
            CLmodel_ = CLmodel(SimDet, V_LV)
    elif isBiV or isFCH:
        if SimDet.get("fch_fe"):
            CLmodel_ = CLmodel_biv(SimDet, V_LV, V_RV, V_LA, V_RA)
        elif SimDet.get("fch_lumped"):
            CLmodel_ = CLmodel_biv(SimDet)
        else:
            CLmodel_ = CLmodel_biv(SimDet, V_LV, V_RV)

    it_ = 0

    csv_path = os.path.join(outputfolder + folderName + "output_PV.csv")
    if MPI.rank(comm_me) == 0:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            headers = [
                "tstep",
                "t",
                "V_LV",
                "P_LV",
                "V_RV",
                "P_RV",
                "V_LA",
                "P_LA",
                "V_RA",
                "P_RA",
                "V_sv",
                "V_sa",
                "V_ad",
                "V_pv",
                "V_pa",
            ]
            writer.writerow(headers)


    while 1:

        if state_obj.cycle > stop_iter:
            break


        if not SimDet.get("fch_lumped") and not SimDet.get("lv_lumped"):
            params = {
                "P_LV": P_LV,
                "V_LV": V_LV,
                "t": state_obj.t,
                "delTat": state_obj.dt.dt,
            }
        else:
            params = {
                "t": state_obj.t,
                "delTat": state_obj.dt.dt,
            }

        if isBiV or isFCH:
            if SimDet.get("fch_fe"):
                params.update(
                    {
                        "P_RV": P_RV,
                        "V_RV": V_RV,
                        "P_LA": P_LA,
                        "V_LA": V_LA,
                        "P_RA": P_RA,
                        "V_RA": V_RA,
                    }
                )
            elif SimDet.get("fch_lumped"):
                pass
            else:
                params.update(
                    {
                        "P_RV": P_RV,
                        "V_RV": V_RV,
                    }
                )

        if isLV or iswaorta:
            if SimDet.get("lv_lumped"):
                V_LV, V_LA = CLmodel_.UpdateLVV(params)
            else:
                V_LV = CLmodel_.UpdateLVV(params)

        elif isBiV or isFCH:
            V_LV, V_RV, *extra = CLmodel_.UpdateLVV(params)
            if SimDet.get("fch_fe") or SimDet.get("fch_lumped"):
                V_LA, V_RA = extra

        print_message = (  # todo: Generalize for all cases
            "t = "
            + str(state_obj.t)
            + " VLV = "
            + str(CLmodel_.V_LV)
            + " Psa = "
            + str(CLmodel_.Psa)
            + " PLV = "
            + str(CLmodel_.PLV)
            + " P_LA = "
            + str(CLmodel_.PLA)
            + " Qmv = "
            + str(CLmodel_.Qmv)
            + " Qav = "
            + str(CLmodel_.Qav)
        )
        if isBiV or isFCH:
            if SimDet.get("fch_fe"):
                print_message += " PLA = " + str(P_LA)
            else:
                print_message += " PLA = " + str(CLmodel_.PLA)
            print_message += "\n"
            print_message += " VRV = " + str(V_RV)
            print_message += " Ppa = " + str(CLmodel_.Ppa)
            print_message += " PRV = " + str(CLmodel_.PRV)
            if SimDet.get("fch_fe"):
                print_message += " PRA = " + str(P_RA)
            else:
                print_message += " PRA = " + str(CLmodel_.PRA)

        printout(print_message, comm_me)

        new_row = [
                state_obj.tstep,
                state_obj.t,
                V_LV,
                P_LV*0.0075,
                V_RV,
                P_RV*0.0075,
                CLmodel_.V_LA,
                "-", #CLmodel_.P_LA,
                CLmodel_.V_RA,
                "-", #P_RA,
                CLmodel_.V_sv,
                CLmodel_.V_sa,
                CLmodel_.V_ad,
                CLmodel_.V_pv,
                CLmodel_.V_pa,
            ]

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)          
            writer.writerow(new_row)

        with open(outputfolder + folderName + "output_PV.txt", "a") as f_PV:
            if MPI.rank(comm_me) == 0:
                if isLV or iswaorta:
                    if SimDet.get("lv_lumped"):
                        f_PV.write(
                            f"{state_obj.tstep}, {CLmodel_.V_LV}, {CLmodel_.PLV} \n"
                        )
                    else:
                        f_PV.write(f"{state_obj.tstep}, {V_LV}, {P_LV} \n")
                elif isBiV or isFCH:
                    if SimDet.get("fch_lumped"):
                        f_PV.write(
                            f"{state_obj.tstep}, {V_LV}, {CLmodel_.PLV}, {V_RV}, {CLmodel_.PRV}, {V_LA}, {CLmodel_.PLA}, {V_RA}, {CLmodel_.PRA} \n"
                        )
                    elif SimDet.get("fch_fe"):
                        pass
                    else:
                        f_PV.write(
                            f"{state_obj.t}, {V_LV}, {P_LV}, {V_RV}, {P_RV}, {CLmodel_.V_LA}, {CLmodel_.PLA}, {CLmodel_.V_RA}, {CLmodel_.PRA}, {CLmodel_.V_sv}, {CLmodel_.V_sa}, {CLmodel_.V_ad}, {CLmodel_.V_pv}, {CLmodel_.V_pa} \n"
                        )

        # Newton's solver
        # import pdb; pdb.set_trace()
        MEmodel_.LVCavitypres.pres = P_LV #LCL

        def Rp(plv, vlv):
            MEmodel_.LVCavitypres.pres = plv
            if SimDet.get("aorta_pres"):
                MEmodel_.AortaCavitypres.pres = CLmodel_.Psa  # * 5e-1
            solver_elas.solvenonlinear()
            v_t = MEmodel_.GetLVV()
            return v_t - vlv

        def Rp_biv(plv, prv, lvvc, rvvc):
            MEmodel_.LVCavitypres.pres = plv
            MEmodel_.RVCavitypres.pres = prv
            solver_elas.solvenonlinear()
            vlv = MEmodel_.GetLVV()
            vrv = MEmodel_.GetRVV()
            return [vlv - lvvc, vrv - rvvc]

        def Rp_fch(plv, prv, pla, pra, lvvc, rvvc, lavc, ravc):
            MEmodel_.LVCavitypres.pres = plv
            MEmodel_.RVCavitypres.pres = prv
            MEmodel_.LACavitypres.pres = pla
            MEmodel_.RACavitypres.pres = pra
            solver_elas.solvenonlinear()
            vlv = MEmodel_.GetLVV()
            vrv = MEmodel_.GetRVV()
            vla = MEmodel_.GetLAV()
            vra = MEmodel_.GetRAV()
            return np.array([vlv - lvvc, vrv - rvvc, vla - lavc, vra - ravc])

        # Create the Newton solver
        from scipy.optimize import (
            newton,
            fsolve,
            #root_scalar,
            bisect,
            root,
            minimize,
        )

        def Rp_call(plv):
            # scale = 1e6
            return Rp(plv, V_LV)  # / scale

        def Rp_biv_call(plrv):
            plv, prv = plrv
            return Rp_biv(plv, prv, V_LV, V_RV)

        def Rp_fch_call(plrva):
            plv, prv, pla, pra = plrva
            return Rp_fch(plv, prv, pla, pra, V_LV, V_RV, V_LA, V_RA)

        if isLV or iswaorta:
            x0 = [P_LV]
        elif isBiV or isFCH:
            if SimDet.get("fch_fe"):
                x0 = [P_LV, P_RV, P_LA, P_RA]
            else:
                x0 = [P_LV, P_RV]

        if isLV or iswaorta:
            if not SimDet.get("lv_lumped"):
                root1, info, ier, msg = fsolve(
                    Rp_call, x0, xtol=1e-4, factor=0.01, full_output=True
                )

        elif isBiV or isFCH:
            if SimDet.get("fch_fe"):
                root1, info, ier, msg = fsolve(
                    Rp_fch_call, x0, xtol=1e-4, factor=0.01, full_output=True
                )
            elif SimDet.get("fch_lumped"):
                pass
            else:
                root1, info, ier, msg = fsolve(
                    Rp_biv_call, x0, xtol=1e-4, factor=0.01, full_output=True
                )

        with open(outputfolder + folderName + "output_nfev.txt", "a") as nfev:
            if (
                MPI.rank(comm_me) == 0
                and not SimDet.get("fch_lumped")
                and not SimDet.get("lv_lumped")
            ):
                nfev.write(
                    f"t = {state_obj.t}, iter = {info['nfev']}, fun = {info['fvec']}, message = {msg} \n"
                )

        if not SimDet.get("fch_lumped") and not SimDet.get("lv_lumped"):
            P_LV = root1[0]

        if isBiV or isFCH:
            if SimDet.get("fch_fe"):
                P_RV = root1[1]
                P_LA = root1[2]
                P_RA = root1[3]
            elif SimDet.get("fch_lumped"):
                pass
            else:
                P_RV = root1[1]

        state_obj.tstep = state_obj.tstep + state_obj.dt.dt
        state_obj.cycle = math.floor(state_obj.tstep / state_obj.BCL)
        state_obj.t = state_obj.tstep - state_obj.cycle * state_obj.BCL

        # Compute local time since activation
        t_init = MEmodel_.activeforms.t_init.vector().get_local()
        local_cycle = np.floor((MEmodel_.t_a.vector().get_local() - t_init)/ state_obj.BCL)
        local_cycle[local_cycle < 0] = 0 # Return zero if cycle is less than zero
        MEmodel_.cycle.vector()[:] = local_cycle
        MEmodel_.t_a.vector()[:] = MEmodel_.t_a.vector().get_local() + state_obj.dt.dt
        MEmodel_.t_since_activation.vector()[:] =  MEmodel_.t_a.vector().get_local()  - t_init - local_cycle * state_obj.BCL
       
        print("Init vector", flush=True)
        print(MEmodel_.activeforms.t_init.vector().get_local(), flush=True)
        print("Cycle", flush=True)
        print(MEmodel_.cycle.vector().get_local(), flush=True)
        #MEmodel_.t_a.vector()[:] = state_obj.t

        isrestart = 0
        state_obj.dt.dt = delTat

        if isPJ:
           # Reset phi and r in EP at end of diastole
           if state_obj.t < state_obj.dt.dt:
               #tstart_arr = [-10]*len(pj_t_nodes)
               coupleEPandPJ_.reset()
               EPmodel_ep.reset()
               if isPJ:
                   EPmodel_pj.reset()

        #    if SimDet.get("CRT", False):
        #         print(f"t = {state_obj.t:.3f} ms, CRT intensity = {EPmodel_ep.fstim_array[-1].iStim}")

           if SimDet.get("CRT", False):
                coupleEPandCRT_.UpdateCRTandEP()   # ensures iStim > 0 at t=0  
        #         for idx in coupleEPandCRT_.fstim_idx:
        #             print(f"t = {state_obj.t:.3f} ms, CRT[{idx}] iStim = {EPmodel_ep.fstim_array[idx].iStim}")

        #    pdb.set_trace()

           if not SimDet.get("lv_lumped") and not ishomo:
                printout("Solving FHN EP", comm_me)
                solver_FHN_ep.solvenonlinear()

                # if isPJ:
                #    # Activate PJ fiber network
                #     if(state_obj.t > SimDet["pacing_timing"][0][0] and \
                #         state_obj.t < SimDet["pacing_timing"][0][0] + SimDet["pacing_timing"][0][1] ):
                #         if("fstim_val_array" in dir(EPmodel_pj)):
                #            EPmodel_pj.fstim_val_array[0] = pj_intensity
                #            printout("pacing", comm_me)#, EPmodel_pj.fstim_val_array)
                #         else:
                #            EPmodel_pj.fstim_array[0].iStim = pj_intensity
                #            printout("pacing", comm_me)#, EPmodel_pj.fstim_array)
                #     else:
                #         if("fstim_val_array" in dir(EPmodel_pj)):
                #             EPmodel_pj.fstim_val_array[0] = 0
                #         else:
                #             EPmodel_pj.fstim_array[0].iStim = 0

                #         printout("not pacing", comm_me)               

                if isPJ:

                    comm = EPmodel_pj.mesh.mpi_comm()
                    rank = comm.Get_rank()
                    size = comm.Get_size()     

                    ploc_pj = SimDet["ploc"]
                    pj_nodes_local = EPmodel_pj.mesh.coordinates()

                    # Build local KDTree
                    pj_tree_local = cKDTree(pj_nodes_local)
                    local_dist, local_idx = pj_tree_local.query(ploc_pj)
                    # Query nearest mesh vertex for each pacing site
                    # distances, pj_node_indices = pj_tree.query(ploc_pj)

                    # Gather all distances and ranks across processes
                    all_dists = comm.allgather(np.atleast_1d(local_dist))
                    all_indices = comm.allgather(np.atleast_1d(local_idx))
                    all_ranks = comm.allgather(np.ones_like(local_idx, dtype=int) * rank)

                    # Find global nearest node for each pacing location
                    all_dists = np.vstack(all_dists)
                    all_indices = np.vstack(all_indices)
                    all_ranks = np.vstack(all_ranks)

                    # Find the global nearest node for each pacing location
                    min_rank_row = np.argmin(all_dists, axis=0)
                    owner_ranks = all_ranks[min_rank_row, np.arange(len(ploc_pj))]
                    owner_local_idx = all_indices[min_rank_row, np.arange(len(ploc_pj))]  

                    if rank == 0:
                        print("owner_ranks:", owner_ranks)
                        print("owner_local_idx:", owner_local_idx)          

                    # Activate PJ fiber network
                    if(state_obj.t > SimDet["pacing_timing"][0][0] and \
                        state_obj.t < SimDet["pacing_timing"][0][0] + SimDet["pacing_timing"][0][1] ):

                        # for idx in pj_node_indices: 
                        for p_idx, (target_rank, local_i) in enumerate(zip(owner_ranks, owner_local_idx)):
                            if rank == target_rank:
                                if "fstim_val_array" in dir(EPmodel_pj) and EPmodel_pj.fstim_val_array is not None:
                                    if 0 <= local_i < len(EPmodel_pj.fstim_array):   
                                        # EPmodel_pj.fstim_val_array[0] = pj_intensity
                                        EPmodel_pj.fstim_val_array[local_i] = pj_intensity                                
                                        # printout("pacing", comm_me)#, EPmodel_pj.fstim_val_array)
                                        printout(f"pacing site {p_idx} via fstim_val_array (rank {rank}, idx {local_i})", comm_me)
                                    else:
                                        print(f"[Rank {rank}] Invalid local_i {local_i} (max {len(EPmodel_pj.fstim_val_array)-1})")
                                elif "fstim_array" in dir(EPmodel_pj) and len(EPmodel_pj.fstim_array) > 0:
                                    # Use pacing-site index instead of mesh node index 
                                    if 0 <= p_idx < len(EPmodel_pj.fstim_array):
                                        EPmodel_pj.fstim_array[p_idx].iStim = pj_intensity
                                        printout(f"pacing site {p_idx} via fstim_array (rank {rank})", comm_me)
                                    else:
                                        print(f"[Rank {rank}] Invalid p_idx {p_idx} (max {len(EPmodel_pj.fstim_array)-1})")
                                else:
                                    print(f"[Rank {rank}] No valid stimulus array found for pacing site {p_idx}")

                            elif state_obj.t >= SimDet["pacing_timing"][0][0] + SimDet["pacing_timing"][0][1]:
                                # for idx in pj_node_indices:
                                for p_idx, (target_rank, local_i) in enumerate(zip(owner_ranks, owner_local_idx)):
                                    if rank == target_rank:
                                        if "fstim_val_array" in dir(EPmodel_pj) and EPmodel_pj.fstim_val_array is not None:
                                            if 0 <= local_i < len(EPmodel_pj.fstim_val_array): 
                                                EPmodel_pj.fstim_val_array[local_i] = 0
                                                printout(f"not pacing site {p_idx} via fstim_val_array (rank {rank}, idx {local_i})", comm_me)
                                            else:
                                                print(f"[Rank {rank}] Invalid local_i {local_i} (max {len(EPmodel_pj.fstim_val_array)-1})")
                                        elif "fstim_array" in dir(EPmodel_pj) and len(EPmodel_pj.fstim_array) > 0:
                                            if 0 <= p_idx < len(EPmodel_pj.fstim_array):
                                                EPmodel_pj.fstim_array[p_idx].iStim = 0
                                                printout(f"not pacing site {p_idx} via fstim_array (rank {rank})", comm_me)
                                            else:
                                                print(f"[Rank {rank}] Invalid p_idx {p_idx} (max {len(EPmodel_pj.fstim_array)-1})")
                                        else:
                                            print(f"[Rank {rank}] No valid stimulus array found for pacing site {p_idx}")



                        #                 else:

                        #                     # EPmodel_pj.fstim_array[0].iStim = pj_intensity
                        #                     EPmodel_pj.fstim_array[local_i].iStim = pj_intensity 
                        #                     pdb.set_trace()                                
                        #                 # printout("pacing", comm_me)#, EPmodel_pj.fstim_array)
                        #                 printout(f"pacing site {p_idx} (rank {rank})", comm_me)
                        #             else:
                                        

                        # else:
                        #     # for idx in pj_node_indices:
                        #     for p_idx, (target_rank, local_i) in enumerate(zip(owner_ranks, owner_local_idx)):
                        #         if rank == target_rank:
                        #             if 0 <= local_i < len(EPmodel_pj.fstim_array): 
                        #                 if("fstim_val_array" in dir(EPmodel_pj)):
                        #                     # EPmodel_pj.fstim_val_array[0] = 0
                        #                     EPmodel_pj.fstim_val_array[local_i] = 0
                        #                 else:
                        #                     # EPmodel_pj.fstim_array[0].iStim = 0
                        #                     EPmodel_pj.fstim_array[local_i].iStim = 0
                        #                 # printout("not pacing", comm_me)
                        #                 printout(f"not pacing site {p_idx} (rank {rank})", comm_me)
                        #             else:
                        #                 print(f"[Rank {rank}] Invalid local_i {local_i} (max {len(EPmodel_pj.fstim_array)-1})")

                
                    #    if (state_obj.t < SimDet["HeartBeatLength"]):
                    #         EPmodel_pj.clamp_variables(lower_bound=-0.05)
                    #         pdb.set_trace()
                    #         w_pj_values = EPmodel_pj.w_ep.vector().get_local()
                    #         w_pj_values = np.maximum(w_pj_values, -0.05)
                    #         EPmodel_pj.w_ep.vector().set_local(w_pj_values)
                    #         EPmodel_pj.w_ep.vector().apply("insert")
                        
                    #         r_pj_values = EPmodel_pj.r.vector().get_local()
                    #         r_pj_values = np.maximum(r_pj_values, -0.05)
                    #         EPmodel_pj.r_.vector().set_local(r_pj_values)
                    #         EPmodel_pj.r_.vector().apply("insert")

                    if("UpdateActivation" in dir(EPmodel_pj)):
                        EPmodel_pj.UpdateActivation()
                    printout("Solving FHN PJ", comm_me)
                    solver_FHN_pj.solvenonlinear()

        else:
            if state_obj.t < state_obj.dt.dt:
                EPmodel_ep.reset()
 
            # Activate EP network
            if(state_obj.t > SimDet["pacing_timing"][0][0] and \
               state_obj.t < SimDet["pacing_timing"][0][0] + SimDet["pacing_timing"][0][1] ):
                for  p in range(0,len(EPmodel_ep.fstim_array)):
                    EPmodel_ep.fstim_array[p].iStim = intensity
                    printout("pacing", comm_me)#, EPmodel_ep.fstim_array)
            else:
                for  p in range(0,len(EPmodel_ep.fstim_array)):
                    EPmodel_ep.fstim_array[p].iStim = 0.0


            # if (state_obj.t < SimDet["HeartBeatLength"]):
            #     EPmodel_ep.clamp_variables(lower_bound=-0.05)
            #    w_ep_values = EPmodel_ep.w_ep.vector().get_local()
            #    w_ep_values = np.maximum(w_ep_values, -0.05)
            #    EPmodel_ep.w_ep.vector().set_local(w_ep_values)
            #    EPmodel_ep.w_ep.vector().apply("insert")

            #    r_ep_values = EPmodel_ep.r_ep.vector().get_local()
            #    r_ep_values = np.maximum(r_ep_values, -0.05)
            #    EPmodel_ep.r_ep.vector().set_local(r_ep_values)
            #    EPmodel_ep.r_ep.vector().apply("insert")


            if not SimDet.get("lv_lumped") and not ishomo:
                printout("Solving FHN EP", comm_me)
                solver_FHN_ep.solvenonlinear()

        if isrestart == 0:
            MEmodel_.UpdateVar()  # For damping
            EPmodel_ep.UpdateVar()
           
            if isPJ:
               EPmodel_pj.UpdateVar()

        # Interpolate phi to mechanics mesh
        potential_ref = EPmodel_ep.interpolate_potential_ep2me_phi(
            V_me=Function(FunctionSpace(MEmodel_.mesh_me, "DG", 0))
        )
        potential_ref.rename("v_ref", "v_ref")

        potential_me.vector()[:] = potential_ref.vector().get_local()[:]

        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
        if MPI.rank(comm_ep) == 0:
            print("UPdating isActiveField and tInitiationField")

        MEmodel_.activeforms.update_activationTime(
            potential_n=potential_me, comm=comm_me
        )

        # Update PJ activation time:
        if isPJ:
            coupleEPandPJ_.UpdatePJandEP()

        if SimDet.get("CRT", False):
            coupleEPandCRT_.UpdateCRTandEP()    

            #probesPJ(EPmodel_pj.getphivar())
            #Nevals = probesPJ.number_of_evaluations()
            #probes_val = probesPJ.array()

            ## broadcast from proc 0 to other processes
            #rank = MPI.rank(comms)
            #probes_val_bcast = probesPJ.array(N=Nevals-1) ## probe will only send to rank =0
            #if(not rank == 0):
            #    probes_val_bcast = np.empty(len(pj_t_nodes))
            #comms.Bcast(probes_val_bcast, root=0)

            #for p in range(0, len(pj_t_nodes)):
            #    phi_pj_val = probes_val_bcast[p]
            #    if(phi_pj_val > 0.9 and tstart_arr[p] < 1.0):
            #        if(tstart_arr[p] < 0):
            #            tstart_arr[p] = 0
            #            EPmodel_ep.fstim_array[p].iStim = intensity
            #        else:
            #            tstart_arr[p] += state_obj.dt.dt
            #    else:
            #        EPmodel_ep.fstim_array[p].iStim = 0.0

        F_n = MEmodel_.GetFmat()
        fstress_DG = project(
            MEmodel_.Getfstress(),
            FunctionSpace(MEmodel_.mesh_me, "DG", 0),
            form_compiler_parameters={"representation": "uflacs"},
        )
        fstress_DG.rename("fstress", "fstress")

        if "probepts" in list(SimDet.keys()):
            probesfstress = Probes(
                x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1)
            )
            probesfstress(fstress_DG)

        Eul_fiber_BiV_DG = project(
            fStrain_uL,
            FunctionSpace(MEmodel_.mesh_me, "DG", 0),
            form_compiler_parameters={"representation": "uflacs"},
        )

        Eul_fiber_BiV_DG.rename("Eff", "Eff")
        if "probepts" in list(SimDet.keys()):
            # x = np.array(SimDet["probepts"])
            probesEul_fiber = Probes(
                x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1)
            )
            probesEul_fiber(Eul_fiber_BiV_DG)

            probesE_circ_BiV = Probes(
                x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1)
            )
            probesE_long_BiV = Probes(
                x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1)
            )
            probesE_radi_BiV = Probes(
                x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1)
            )
        # pdb.set_trace()
        # postprocess and write
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
        export.writePV(MEmodel_, state_obj.tstep, CLmodel = CLmodel_)

        if isLV:
            export.writeQ(MEmodel_, [CLmodel_.Qsa, CLmodel_.Qad, CLmodel_.Qsv, 
                                     CLmodel_.Qmv, CLmodel_.Qav], state_obj.tstep)
            export.writeP(MEmodel_, np.array([CLmodel_.Psa, CLmodel_.Pad, CLmodel_.Psv, 
                                              CLmodel_.PLA, CLmodel_.PLV])*0.0075, state_obj.tstep)
            export.writeV(MEmodel_, np.array([CLmodel_.V_sa, CLmodel_.V_ad, CLmodel_.V_sv, 
                                              CLmodel_.V_LA, CLmodel_.V_LV]), state_obj.tstep)


        if isBiV:
            export.writeQ(MEmodel_, np.array([CLmodel_.Qsa, CLmodel_.Qad, CLmodel_.Qsv, CLmodel_.Qmv, CLmodel_.Qav,
                                              CLmodel_.Qtv, CLmodel_.Qpvv, CLmodel_.Qpa, CLmodel_.Qpv
                                              ]), state_obj.tstep)

            export.writeP(MEmodel_, np.array([CLmodel_.Psa, CLmodel_.Pad, CLmodel_.Psv, CLmodel_.PLA, CLmodel_.PLV,
                                              CLmodel_.Ppa, CLmodel_.Ppv, CLmodel_.PRV,
                                              CLmodel_.PRA])*0.0075, state_obj.tstep)

            export.writeV(MEmodel_, np.array([CLmodel_.V_sa, CLmodel_.V_ad, CLmodel_.V_sv, CLmodel_.V_LA, CLmodel_.V_LV,
                                              CLmodel_.V_pa, CLmodel_.V_pv, CLmodel_.V_RV,
                                              CLmodel_.V_RA]), state_obj.tstep)


        if cnt % SimDet["writeStep"] == 0.0:
            export.writetpt(MEmodel_, state_obj.tstep)
            export.hdf.write(MEmodel_.GetDisplacement(), "ME/u", writecnt)
            export.hdf.write(potential_ref, "ME/potential_ref", writecnt)
            export.hdf.write(MEmodel_.GetSActive(), "ME/Sactive", writecnt)
            export.hdf.write(MEmodel_.Get_t_a(), "ME/t_a", writecnt)
            export.hdf.write(MEmodel_.Get_t_init(), "ME/t_init", writecnt)
            export.hdf.write(MEmodel_.Get_isActive(), "ME/isActive", writecnt)
            export.hdf.write(MEmodel_.Get_local_cycle(), "ME/cycle", writecnt)
            export.hdf.write(MEmodel_.Get_t_since_act(), "ME/t_since_act", writecnt)
            export.hdf.write(Eul_fiber_BiV_DG, "ME/Eff", writecnt)
            export.hdf.write(fstress_DG, "ME/fstress", writecnt)
            export.hdf.write(MEmodel_.GetP(), "ME/imp_constraint", writecnt)

            export.hdf.write(EPmodel_ep.getphivar(), "EP/phi", writecnt)
            export.hdf.write(EPmodel_ep.getrvar(), "EP/r", writecnt)
            export.hdf.write(potential_ref, "EP/potential_ref", writecnt)

            if isPJ:
                export.hdf.write(EPmodel_pj.getphivar(), "PJ/phi", writecnt)
                export.hdf.write(EPmodel_pj.getrvar(), "PJ/r", writecnt)

            writecnt += 1

        if "probepts" in list(SimDet.keys()):
            fIMP = probesIMP.array()
            fIMP2 = probesIMP2.array()
            fIMP3 = probesIMP3.array()
            fStress = probesfstress.array()
            fStrain_vals = probesEul_fiber.array()
            E_circ_BiV = probesE_circ_BiV.array()
            E_long_BiV = probesE_long_BiV.array()
            E_radi_BiV = probesE_radi_BiV.array()

            export.writeIMP(MEmodel_, state_obj.tstep, fIMP)
            export.writeIMP2(MEmodel_, state_obj.tstep, fIMP2)
            export.writeIMP3(MEmodel_, state_obj.tstep, fIMP3)
            export.writefStress(MEmodel_, state_obj.tstep, fStress)
            export.writefStrain(MEmodel_, state_obj.tstep, fStrain_vals)
            export.writeCStrain(MEmodel_, state_obj.tstep, E_circ_BiV)
            export.writeLStrain(MEmodel_, state_obj.tstep, E_long_BiV)
            export.writeRStrain(MEmodel_, state_obj.tstep, E_radi_BiV)

        cnt += 1

        if(state_obj.tstep % state_obj.BCL == 0) :
            export.dump_restart_file(CLmodel_)


# def createEPmodel(IODet, SimDet):

#     directory_ep = IODet["directory_ep"]
#     outputfolder = IODet["outputfolder"]
#     folderName = IODet["folderName"] + IODet["caseID"] + "/"
#     delTat = SimDet["dt"]

#     if "casename_ep" in IODet:
#         casename = IODet["casename_ep"]
#     else:
#         casename = IODet["casename"]

#     casename_marked = IODet["casename_ep_marked"]  

#     if "isFCH" in list(SimDet.keys()):
#         isFCH = SimDet["isFCH"]
#     else:
#         isFCH = False
#     if "iswaorta" in list(SimDet.keys()):
#         iswaorta = SimDet["iswaorta"]
#     else:
#         iswaorta = False  # Default

#     #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
#     # Read EP data from HDF5 Files
#     mesh_ep = Mesh()
#     comm_common = mesh_ep.mpi_comm()

#     meshfilename_ep = directory_ep + casename + ".hdf5"
#     f = HDF5File(comm_common, meshfilename_ep, "r")
#     f.read(mesh_ep, casename, False)
#     if isFCH:
#         mesh_ep.scale(6.5e-2)
#     elif iswaorta:
#         mesh_ep.scale(1.1)

#     # meshfilename_ep_marked = directory_ep + casename_marked + ".hdf5"
#     # f = HDF5File(comm_common, meshfilename_ep_marked, "r")
#     # f.read(mesh_ep_marked, casename_marked, False)
#     # if isFCH:
#     #     mesh_ep_marked.scale(6.5e-2)
#     # elif iswaorta:
#     #     mesh_ep_marked.scale(1.1)        

#     File(outputfolder + folderName + "mesh_ep.pvd") << mesh_ep

#     facetboundaries_ep = MeshFunction("size_t", mesh_ep, 2)
#     f.read(facetboundaries_ep, casename + "/" + "facetboundaries")

#     matid_ep = MeshFunction("size_t", mesh_ep, mesh_ep.topology().dim())
#     AHAid_ep = MeshFunction("size_t", mesh_ep, mesh_ep.topology().dim())
#     # baseid_ep = MeshFunction("size_t", mesh_ep, mesh_ep.topology().dim())

#     if f.has_dataset(casename + "/" + "matid"):
#         if SimDet.get("function_matid"):
#             VQuadelem = FiniteElement(
#                 "DG", mesh_ep.ufl_cell(), degree=0, quad_scheme="default"
#             )
#             matid_FS = FunctionSpace(mesh_ep, VQuadelem)
#             matid_func = dolfin.Function(matid_FS)
#             for cell in cells(mesh_ep):
#                 matid_ep[cell.index()] = round(matid_func(cell.midpoint()))
#         else:
#             f.read(matid_ep, casename + "/" + "matid")
#     else:
#         matid_ep.set_all(0)

#     if f.has_dataset(casename + "/" + "AHAid"):
#         f.read(AHAid_ep, casename + "/" + "AHAid")
#     else:
#         AHAid_ep.set_all(0)    

#     # if f.has_dataset(casename + "/" + "baseid"):
#     #     if SimDet.get("function_baseid"):
#     #         VQuadelem = FiniteElement("DG", mesh_ep.ufl_cell(), degree=0, quad_scheme="default")
#     #         base_FS = FunctionSpace(mesh_ep, VQuadelem)
#     #         baseid_func = dolfin.Function(baseid_FS)
#     #         for cell in cells(mesh_ep):
#     #             baseid_ep[cell.index()] = round(baseid_func(cell.midpoint()))
#     #     else:
#     #         f.read(baseid_ep, casename + "/" + "baseid")
#     # else:
#     #     baseid_ep.set_all(0)        

#     mesh_ep_marked = Mesh()
#     comm_common_marked = mesh_ep_marked.mpi_comm()

#     meshfilename_ep_marked = directory_ep + casename_marked + ".hdf5"
#     f_marked = HDF5File(comm_common_marked, meshfilename_ep_marked, "r")
#     f_marked.read(mesh_ep_marked, casename_marked, False)

#     # Load ischemia mask for EP mesh
#     EP_ischemia = load_ischemia_mask(mesh_ep_marked, directory_ep, casename_marked)

#     deg_ep = 4

#     Quadelem_ep = FiniteElement(
#         "Quadrature", mesh_ep.ufl_cell(), degree=deg_ep, quad_scheme="default"
#     )
#     Quadelem_ep._quad_scheme = "default"
#     Quad_ep = FunctionSpace(mesh_ep, Quadelem_ep)

#     if "fiber_fspace" in list(SimDet.keys()) and "fiber_fspace_deg" in list(
#         SimDet.keys()
#     ):
#         VQuadelem_ep = VectorElement(
#             SimDet["fiber_fspace"],
#             mesh_ep.ufl_cell(),
#             degree=SimDet["fiber_fspace_deg"],
#             quad_scheme="default",
#         )
#         VQuadelem_ep._quad_scheme = "default"
#     else:
#         VQuadelem_ep = VectorElement(
#             "Quadrature", mesh_ep.ufl_cell(), degree=deg_ep, quad_scheme="default"
#         )
#         VQuadelem_ep._quad_scheme = "default"

#     fiberFS_ep = FunctionSpace(mesh_ep, VQuadelem_ep)

#     f0_ep = Function(fiberFS_ep)
#     s0_ep = Function(fiberFS_ep)
#     n0_ep = Function(fiberFS_ep)

#     if SimDet["DTI_EP"] is True:
#         f.read(f0_ep, casename + "/" + "eF_DTI")
#         f.read(s0_ep, casename + "/" + "eS_DTI")
#         f.read(n0_ep, casename + "/" + "eN_DTI")
#     else:
#         f.read(f0_ep, casename + "/" + "eF")
#         f.read(s0_ep, casename + "/" + "eS")
#         f.read(n0_ep, casename + "/" + "eN")

#     f.close()

#     # --------------------------------------------------------
#     # LOAD ISCHEMIA MASK from the *_marked.hdf5 file
#     # --------------------------------------------------------
#     ischemia_mask_ep = MeshFunction("size_t", mesh_ep, mesh_ep.topology().dim(), 0)

#     if casename_marked is not None:
#         maskfile = os.path.join(directory_ep, casename_marked + ".hdf5")
#         if os.path.exists(maskfile):
#             try:
#                 fm = HDF5File(comm_common, maskfile, "r")
#                 fm.read(ischemia_mask_ep, casename_marked + "/ischemia_mask")
#                 fm.close()
#                 print(f"Loaded EP ischemia_mask from {maskfile}")
#             except Exception as e:
#                 print("⚠️ Failed to read EP ischemia_mask, using all-zero mask.", e)
#         else:
#             print(f"⚠️ No marked EP file {maskfile}, using all-zero mask.")    

#     comm_ep = mesh_ep.mpi_comm()

#     # Define state variables
#     #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
#     state_obj = State_Variables(comm_ep, SimDet)
#     state_obj.dt.dt = delTat
#     #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

#     if "d_iso" in list(SimDet.keys()):
#         d_iso = SimDet["d_iso"]
#     else:
#         d_iso = 0.02

#     if "d_ani_factor" in list(SimDet.keys()):
#         d_ani_factor = SimDet["d_ani_factor"]
#     else:
#         d_ani_factor = 0.02

#     if "ani_factor" in list(SimDet.keys()):
#         ani_factor = SimDet["ani_factor"]
#     else:
#         ani_factor = 1000.0

#     if "CRT" in list(SimDet.keys()):
#         CRT = SimDet["CRT"]
#     else:
#         CRT = False

#     if "CRT_current_intensity" in list(SimDet.keys()):
#         CRT_current_intensity = SimDet["CRT_current_intensity"]
#     else:
#         CRT_current_intensity = 0.0        

#     if "CRT_pos" in list(SimDet.keys()):
#         CRT_pos = SimDet["CRT_pos"]
#     else:
#         CRT_pos = [0.0,0.0,0.0]

#     if "CRT_pacing_timing" in list(SimDet.keys()):
#         CRT_pacing_timing = SimDet["CRT_pacing_timing"]
#     else:
#         CRT_pacing_timing = [[0.0, 0.0]]        


#     EPparams = {
#         "EPmesh": mesh_ep,
#         "deg": 4,
#         "matid": matid_ep,
#         "facetboundaries": facetboundaries_ep,
#         "f0": f0_ep,
#         "s0": s0_ep,
#         "n0": n0_ep,
#         "state_obj": state_obj,
#         "d_iso": d_iso,
#         "d_ani": d_ani_factor,
#         "ani_factor": ani_factor,
#         # "ploc": SimDet["ploc"],
#         "AHAid": AHAid_ep,
#         # "matid": matid_ep,
#         "Ischemia": SimDet["Ischemia"],
#         "Ischemia_mask": ischemia_mask_ep,
#         "abs_tol": SimDet["abs_tol"],
#         "rel_tol": SimDet["rel_tol"],
#         "CRT": CRT,  
#         "CRT_current_intensity": CRT_current_intensity,
#         "CRT_pos": CRT_pos,   
#         "CRT_pacing_timing": CRT_pacing_timing,    
#     }

#     if "ploc" in list(SimDet.keys()):
#         EPparams.update({"ploc": SimDet["ploc"]})

#     # if "isPJ" in list(SimDet.keys()):
#     #     if SimDet["isPJ"] and "pj_tnodes" in list(SimDet.keys()):
#     #         EPparams.update({"ploc": SimDet["pj_tnodes"]})

#     if "isPJ" in SimDet and SimDet["isPJ"] and "pj_tnodes" in SimDet:
#         pj_tnodes = SimDet["pj_tnodes"]
#     else:
#         pj_tnodes = []

#     crt_pos    = SimDet.get("CRT_pos", []) if SimDet.get("CRT", False) else []

#     # Merge both pacing sets into a single ploc list
#     ploc_all = list(pj_tnodes) + list(crt_pos)

#     # Record counts for later index separation
#     nPJ  = len(pj_tnodes)
#     nCRT = len(crt_pos)

#     # --- Update EPparams ---
#     EPparams.update({
#         "ploc": ploc_all,
#         "nPJ":  nPJ,
#         "nCRT": nCRT
#     })
            

#         # if SimDet["isPJ"] and "PJ_current_intensity" in list(SimDet.keys()):
#     EPparams.update({"current_intensity": SimDet["current_intensity"]})

#     # Attach to params for use by EPModel
#     EPparams["ischemia_mask"] = EP_ischemia


#     if "Ischemia" in list(SimDet.keys()):
#         EPparams.update({"Ischemia": SimDet["Ischemia"]})

#     if "pacing_timing" in list(SimDet.keys()):
#         EPparams.update({"pacing_timing": SimDet["pacing_timing"]})

#     # Define EP model and solver
#     EPmodel_ = EPmodel(EPparams)

#     return EPmodel_, state_obj, EPparams

# def createPJmodel(IODet, SimDet):

#     outputfolder = IODet["outputfolder"]
#     folderName = IODet["folderName"] + IODet["caseID"] + "/"
#     directory_pj = IODet["directory_pj"]
#     casename = IODet["casename_pj"]
#     delTat = SimDet["dt"]

#     # Read EP data from HDF5 Files
#     mesh_pj = Mesh()
#     comm_pj = mesh_pj.mpi_comm()

#     meshfilename_pj = directory_pj + casename + ".hdf5"
#     f = HDF5File(comm_pj, meshfilename_pj, "r")
#     f.read(mesh_pj, casename, False)
#     File(outputfolder + folderName + "mesh_pj.pvd") << mesh_pj

#     AHAid_pj = MeshFunction(
#         "size_t", mesh_pj, 1, mesh_pj.domains()
#     )
#     AHAid_pj.set_all(0)

#     matid_pj = MeshFunction(
#         "size_t", mesh_pj, 1, mesh_pj.domains()
#     )
#     matid_pj.set_all(0)
#     f.read(matid_pj, casename + "/" + "matid")

#     # Define state variables
#     #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
#     state_obj = State_Variables(comm_pj, SimDet)
#     state_obj.dt.dt = delTat
#     pj_intensity = SimDet["PJ_current_intensity"]
    
#     # Define EP model and solver
#     EPparams_pj = {
#         "EPmesh": mesh_pj,
#         "deg": 4,
#         "state_obj": state_obj,
#         "d_iso": SimDet["d_pj"],
#         "ploc": SimDet["ploc"],
#         "AHAid": AHAid_pj,
#         "matid": matid_pj,
#         "ploc_mode":SimDet["ploc_mode"],
#         "lbbb":SimDet["lbbb"],
#         "lbbb_delay": SimDet["lbbb_delay"], 
#         "lbbb_location": SimDet["lbbb_location"], 
#         "abs_tol": SimDet["abs_tol"],
#         "rel_tol": SimDet["rel_tol"],            
#     }

#     if("d_iso_pj" in list(SimDet.keys())):
#         EPparams_pj.update({"d_iso_pj": SimDet["ploc"]})


#     # Define EP model and solver
#     EPmodel_ = EPmodel(EPparams_pj)

#     return EPmodel_, state_obj, EPparams_pj


#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
if __name__ == "__main__":
    print("Testing...")
    run_BiV_TimedGuccione(IODet=IODetails, SimDet=SimDetails)

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
