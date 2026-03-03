import dolfin as df
from .postprocessdatalib2 import *
from ..mechanics.forms_MRC2 import Forms
from ..utils.oops_objects_MRC2 import State_Variables
from ..utils.oops_objects_MRC2 import lv_mesh as lv_mechanics_mesh
from ..utils.oops_objects_MRC2 import biventricle_mesh as biv_mechanics_mesh
from ..mechanics.MEmodel3 import MEmodel
import matplotlib
matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import collections  # <-- This line is required
from matplotlib import pylab as plt
# from mpi4py import MPI as pyMPI
from mpi4py import MPI
import pdb


def postprocessdata(IODet, SimDet, cycle=None):
    directory = IODet["outputfolder"] + "/"
    casename = IODet["caseID"]
    BCL = SimDet["HeartBeatLength"]
    isLV = SimDet["isLV"]

    if cycle is None:
        cycle = SimDet["closedloopparam"]["stop_iter"]

    for ncycle in range(cycle - 1, cycle):
        filename = directory + casename + "/" + "BiV_PV.txt"
        homo_tptt, homo_LVP, homo_LVV, homo_RVP, homo_RVV, homo_Qmv = extract_PV(filename, BCL, ncycle, SimDet)
        # pdb.set_trace()
        if SimDet["isLV"]:
            filename = directory + casename + "/" + "BiV_Q.txt"
            (   homo_tptt,
                homo_Qao,
                homo_Qmv,
                homo_Qper,
                homo_Qla,
                homo_Qlad,
                homo_Qlcx,
                Q_lvad,
            ) = extract_Q(filename, BCL, ncycle, SimDet)
        else:
            filename = directory + casename + "/" + "BiV_Q.txt"
            (   homo_tptt,
                homo_Qao,
                homo_Qmv,
                Qsa,
                Qsv,
                homo_Qper,
                homo_Qla,
                homo_Qlad,
                homo_Qlcx,
                Q_lvad,
            ) = extract_Q(filename, BCL, ncycle, SimDet)
     


        filename = directory + casename + "/" + "BiV_P.txt"
        homo_tptt, homo_Pven, homo_LVPP, homo_Part, homo_PLA, Ppv, PRV, Ppa, PRA = extract_P(filename, BCL, ncycle, SimDet)  
        # tpt, Psv, PLV, Psa, PLA, Ppv, PRV, Ppa, PRA = extract_P(filename, BCL, ncycle, SimDet)           
        
        filename = directory + casename + "/" + "BiV_IMP_InC.txt"
        homo_tpt_IMP, homo_IMP = extract_probe(filename, BCL, ncycle)

        filename = directory + casename + "/" + "BiV_fiberStrain.txt"
        homo_tpt_Eff, homo_Eff = extract_probe(filename, BCL, ncycle)

        filename = directory + casename + "/" + "BiV_fiberStress.txt"
        homo_tpt_Sff, homo_Sff = extract_probe(filename, BCL, ncycle)

        ESP, ESV = extractESP(homo_LVP, homo_LVV)

        EDP, EDV = extractEDP(homo_LVP, homo_LVV)

        # SBP = max(homo_Part) * 0.0075
        # DBP = min(homo_Part) * 0.0075
        SBP = max(homo_Part)
        DBP = min(homo_Part)

        if max(homo_LVP) < 25:
            EF = 0.0
        else:
            EF = (max(homo_LVV) - min(homo_LVV)) / max(homo_LVV)


        print(filename)
        print(
            (
                "EF = ",
                EF ,
                " EDV = ",
                max(homo_LVV),
                " ESV = ",
                min(homo_LVV),
                " EDP = ",
                EDP,
                " SBP = ",
                SBP,
                " DBP = ",
                DBP,
            )
        )

        print(("Peak LV pressure = ", max(homo_LVP)))

        # homo_directory = directory + casename + "/"

        # tpt_array = readtpt(homo_directory + "tpt.txt")
        # ind = np.where((tpt_array > (ncycle) * BCL) * (tpt_array < (ncycle + 1) * BCL))
        # tpt = tpt_array[ind]

        ## Get Point cloud for probing
        # ptcloud, radialpos, vtkradialpos = getpointclouds(homo_directory, clipoffset=5e-1, npts=10000)
        ptcloud, radialpos, vtkradialpos = getpointclouds(
            directory + casename + "/", clipoffset=1e-5, npts=10000
        )
        # ## Get Point cloud for probing
        # # ptcloud, radialpos, vtkradialpos = getpointclouds(homo_directory, clipoffset=5e-1, npts=10000)
        # ptcloud, radialpos, vtkradialpos = getpointclouds(
        #     directory + casename + "/", clipoffset=1e-5, npts=10000
        # )

        vtk_py.writeXMLPData(vtkradialpos, casename + ".vtp")
        # vtk_py.writeXMLPData(vtkradialpos, casename + ".vtp")

        # Get transmural variation of IMP
        index = find_nearest(
            tpt, homo_tptt[np.argmax(homo_LVPP)]
        )  # Find ID correspond to peak LV pressure
        imp = probeqty(homo_directory, "ME/imp_constraint", ptcloud, ind, index)
        imp = imp * 0.0075
        # # Get transmural variation of IMP
        # index = find_nearest(
        #     tpt, homo_tptt[np.argmax(homo_LVPP)]
        # )  # Find ID correspond to peak LV pressure
        # imp = probeqty(homo_directory, "ME/imp_constraint", ptcloud, ind, index)
        # imp = imp * 0.0075

        ## Get transmural variation of WD
        Sff = probetimeseries(homo_directory, "ME/fstress", ptcloud, ind, "DG", 0)
        Eff = probetimeseries(homo_directory, "ME/Eff", ptcloud, ind, "DG", 0)
        WD = np.array(
            [
                -1.0 * np.trapz(Sff[:, i] * 0.0075, Eff[:, i])
                for i in range(0, len(Sff[1, :]))
            ]
        )
        # ## Get transmural variation of WD
        # Sff = probetimeseries(homo_directory, "ME/fstress", ptcloud, ind, "DG", 0)
        # Eff = probetimeseries(homo_directory, "ME/Eff", ptcloud, ind, "DG", 0)
        # WD = np.array(
        #     [
        #         -1.0 * np.trapz(Sff[:, i] * 0.0075, Eff[:, i])
        #         for i in range(0, len(Sff[1, :]))
        #     ]
        # )

        # Convert to vtp flie
        for i in range(0, len(Sff[:, 1])):
            pdata = vtk.vtkPolyData()
            pdata.DeepCopy(vtkradialpos)
            Sff_VTK_data = numpy_support.numpy_to_vtk(
                num_array=0.0075 * Sff[i, :].ravel(),
                deep=True,
                array_type=vtk.VTK_FLOAT,
            )
            Sff_VTK_data.SetName("fstress_")
            pdata.GetPointData().AddArray(Sff_VTK_data)
            Eff_VTK_data = numpy_support.numpy_to_vtk(
                num_array=Eff[i, :].ravel(), deep=True, array_type=vtk.VTK_FLOAT
            )
            Eff_VTK_data.SetName("Eff_")
            pdata.GetPointData().AddArray(Eff_VTK_data)
            WD_VTK_data = numpy_support.numpy_to_vtk(
                num_array=WD.ravel(), deep=True, array_type=vtk.VTK_FLOAT
            )
            WD_VTK_data.SetName("WD_")
            pdata.GetPointData().AddArray(WD_VTK_data)
            # vtk_py.writeXMLPData(pdata, casename+"fstress"+str(i)+".vtp")
        # # Convert to vtp flie
        # for i in range(0, len(Sff[:, 1])):
        #     pdata = vtk.vtkPolyData()
        #     pdata.DeepCopy(vtkradialpos)
        #     Sff_VTK_data = numpy_support.numpy_to_vtk(
        #         num_array=0.0075 * Sff[i, :].ravel(),
        #         deep=True,
        #         array_type=vtk.VTK_FLOAT,
        #     )
        #     Sff_VTK_data.SetName("fstress_")
        #     pdata.GetPointData().AddArray(Sff_VTK_data)
        #     Eff_VTK_data = numpy_support.numpy_to_vtk(
        #         num_array=Eff[i, :].ravel(), deep=True, array_type=vtk.VTK_FLOAT
        #     )
        #     Eff_VTK_data.SetName("Eff_")
        #     pdata.GetPointData().AddArray(Eff_VTK_data)
        #     WD_VTK_data = numpy_support.numpy_to_vtk(
        #         num_array=WD.ravel(), deep=True, array_type=vtk.VTK_FLOAT
        #     )
        #     WD_VTK_data.SetName("WD_")
        #     pdata.GetPointData().AddArray(WD_VTK_data)
        #     # vtk_py.writeXMLPData(pdata, casename+"fstress"+str(i)+".vtp")

        ## Get Ecc
        Ecc = probetimeseries(homo_directory, "ME/Ecc", ptcloud, ind, "DG", 0)
        peakEcc = np.max(np.abs(np.mean(Ecc, axis=1) * 100))
        print(("Peak Ecc = ", peakEcc))
        # ## Get Ecc
        # Ecc = probetimeseries(homo_directory, "ME/Ecc", ptcloud, ind, "DG", 0)
        # peakEcc = np.max(np.abs(np.mean(Ecc, axis=1) * 100))
        # print(("Peak Ecc = ", peakEcc))

        # Get Ell
        Ell = probetimeseries(homo_directory, "ME/Ell", ptcloud, ind, "DG", 0)
        peakEll = np.max(np.abs(np.mean(Ell, axis=1) * 100))
        print(("Peak Ell = ", peakEll))
        # # Get Ell
        # Ell = probetimeseries(homo_directory, "ME/Ell", ptcloud, ind, "DG", 0)
        # peakEll = np.max(np.abs(np.mean(Ell, axis=1) * 100))
        # print(("Peak Ell = ", peakEll))

        # np.savez(
        #     directory + casename + "/" + casename + ".npz",
        #     homo_tptt=homo_tptt,
        #     homo_LVP=homo_LVP,
        #     homo_LVV=homo_LVV,
        #     homo_Qmv=homo_Qmv,
        #     homo_Qao=homo_Qao,
        #     homo_Qper=homo_Qper,
        #     homo_Qla=homo_Qla,
        #     homo_Qlad=homo_Qlad,
        #     homo_Pven=0.0075 * homo_Pven,
        #     homo_LVPP=0.0075 * homo_LVPP,
        #     homo_Part=0.0075 * homo_Part,
        #     homo_PLA=0.0075 * homo_PLA,
        #     homo_tpt_IMP=0.0075 * homo_tpt_IMP,
        #     homo_IMP=homo_IMP,
        #     homo_tpt_Eff=homo_tpt_Eff,
        #     homo_Eff=homo_Eff,
        #     homo_tpt_Sff=homo_tpt_Sff,
        #     homo_Sff=homo_Sff,
        #     ESP=ESP,
        #     ESV=ESV,
        #     EDP=EDP,
        #     EDV=EDV,
        #     SBP=SBP,
        #     DBP=DBP,  # Qtotal       = Qtotal,\
        #     imp=imp,
        #     radialpos=radialpos,
        #     Eff=Eff,
        #     Sff=Sff,
        #     WD=WD,
        #     Ecc=Ecc,
        #     Ell=Ell,
        #     BCL=BCL,
        #     tpt=tpt,
        #     ncycle=ncycle,
        # )
        # np.savez(
        #     directory + casename + "/" + casename + ".npz",
        #     homo_tptt=homo_tptt,
        #     homo_LVP=homo_LVP,
        #     homo_LVV=homo_LVV,
        #     homo_Qmv=homo_Qmv,
        #     homo_Qav=homo_Qav,
        #     homo_Qsa=homo_Qsa,
        #     homo_Qsv=homo_Qsv,
        #     homo_Pven=0.0075 * homo_Pven,
        #     homo_LVPP=0.0075 * homo_LVPP,
        #     homo_Part=0.0075 * homo_Part,
        #     homo_PLA=0.0075 * homo_PLA,
        #     homo_tpt_IMP=0.0075 * homo_tpt_IMP,
        #     ESP=ESP,
        #     ESV=ESV,
        #     EDP=EDP,
        #     EDV=EDV,
        #     SBP=SBP,
        #     DBP=DBP,  
        #     tpt=tpt,
        #     ncycle=ncycle,
        # )

def compute_activation(IODet, SimDet, cycle=None):

    mesh = df.Mesh()
    hdf = df.HDF5File(mesh.mpi_comm(), IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5", "r",)
    hdf.read(mesh, "EP/mesh", False)

    phi_arr = extractvtk(IODet["outputfolder"] + "/" + IODet["caseID"], "EP/phi", "CG", 1, IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "EP_" + "phi", "phi", group="EP", iswrite=False,)

    V_thres = 0.9
    time_act = df.Function(df.FunctionSpace(mesh, "CG", 1))
    time_act_vec = -1 * np.ones(len(time_act.vector()[:]))


    dt = SimDet["dt"]
    t = dt    
    write_t = SimDet["writeStep"]

    for phi in phi_arr:
        phi_vec = phi.sub(0).vector().get_local()[::3]
        for idx, (time_act_vec_, phi_vec_) in enumerate(zip(time_act_vec, phi_vec)):
            if phi_vec_ > V_thres and time_act_vec_ == -1:
                time_act_vec[idx] = t

        t += dt * write_t

    time_act.vector()[:] = time_act_vec
    time_act.rename("Activation Time", "Activation Time")

    # early_time_act_vec = time_act_vec - min(time_act_vec)
    # t_90_myo = np.percentile(early_time_act_vec,90)

    valid_mask = time_act_vec != -1
    valid_times = time_act_vec[valid_mask]

    # pdb.set_trace()
    # early_time_act_vec = valid_times - min(valid_times)
    # t_90_myo = np.percentile(early_time_act_vec, 90)

    t_90_myo = np.percentile(time_act_vec,90)    
    print("90% Activation time myocardium:", t_90_myo, "[ms]")

    act_outdirectory = os.path.join( IODet["outputfolder"], IODet["caseID"], "activation")
    if not os.path.exists(act_outdirectory):
        os.mkdir(act_outdirectory)
    File_act = df.File(os.path.join(act_outdirectory, "act.pvd"))

    if np.all(valid_mask == False):
        time_act.vector()[:]=time_act.vector()[:]
    else: 
        time_act.vector()[:]=time_act.vector()[:]-min(valid_times)

    File_act << time_act

    if not SimDet["isPJ"]:
        with open(act_outdirectory+"/activation_summary_myo.txt", "w") as f:
            f.write("90% Activation Time [ms]\n")
            f.write(f"90% Activation time myocardium:\t{t_90_myo:.3f}[ms]\n")  


    # PJ activation
    if SimDet["isPJ"]:

        mesh_pj = df.Mesh()
        hdf = df.HDF5File(mesh_pj.mpi_comm(),IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5","r",)
        hdf.read(mesh_pj, "PJ/mesh", False)
        pj_phi_arr = extractvtk(IODet["outputfolder"] + "/" + IODet["caseID"], "PJ/phi", "CG", 1, IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "PJ_" + "phi", "phi", group="PJ", iswrite=False,)

        # pj_V_thres = 0.9
        pj_time_act = df.Function(df.FunctionSpace(mesh_pj, "CG", 1))
        pj_time_act_vec = -1 * np.ones(len(pj_time_act.vector()[:]))
        t = dt
        for pj_phi in pj_phi_arr:
            pj_phi_vec = pj_phi.sub(0).vector().get_local()[::3]
            for idx, (pj_time_act_vec_, pj_phi_vec_) in enumerate(zip(pj_time_act_vec, pj_phi_vec)):
                if pj_phi_vec_ > V_thres and pj_time_act_vec_ == -1:
                    pj_time_act_vec[idx] = t

            t += dt * write_t

        # ---- shift only valid values ----
        valid = pj_time_act_vec[pj_time_act_vec >= 0]

        if len(valid) > 0:
            min_valid = np.min(valid)

            # create shifted array but keep -1 unchanged
            shifted = np.copy(pj_time_act_vec)
            mask_valid = shifted >= 0
            shifted[mask_valid] -= min_valid

            pj_time_act.vector()[:] = shifted
            t_90_pj = np.percentile(valid, 90)

        else:
            # no PJ activations at all
            pj_time_act.vector()[:] = pj_time_act_vec
            t_90_pj = -1

        # pj_time_act.vector()[:] = pj_time_act_vec - min(pj_time_act_vec)
        pj_time_act.rename("PJ Activation Time", "PJ Activation Time")
        # pdb.set_trace()

        # t_90_pj = np.percentile(pj_time_act_vec,90)
        # print("90% Activation time purkinje:", t_90_pj, "[ms]")

        File_act_pj = df.File(os.path.join(act_outdirectory, "pj_act.pvd"))
        File_act_pj << pj_time_act

        with open(act_outdirectory+"/activation_summary.txt", "w") as f:
            f.write("90% Activation Time [ms]\n")
            f.write(f"90% Activation time purkinje:\t{t_90_pj:.3f}[ms]\n")
            f.write(f"90% Activation time myocardium:\t{t_90_myo:.3f}[ms]\n")        


def normalize_directionalbasis(eC0, eL0, eR0, mesh, deg):

    #eC0 = Mesh_obj.eC0
    #eL0 = Mesh_obj.eL0
    #eR0 = Mesh_obj.eR0

    eC0_normalized = eC0 / df.sqrt(df.inner(eC0, eC0))
    eL0_normalized = eL0 / df.sqrt(df.inner(eL0, eL0))
    eR0_normalized = eR0 / df.sqrt(df.inner(eR0, eR0))

    eC0_normalized = (
        df.project(
            eC0_normalized,
            df.VectorFunctionSpace(mesh, "DG", 0),
            form_compiler_parameters={
                "representation": "uflacs",
                "quadrature_degree": deg,
            },
        )
        .vector()
        .get_local()
    )
    isnan_eC0_normalized = np.argwhere(np.isnan(eC0_normalized)).flatten()

    eL0_normalized = (
        df.project(
            eL0_normalized,
            df.VectorFunctionSpace(mesh, "DG", 0),
            form_compiler_parameters={
                "representation": "uflacs",
                "quadrature_degree": deg,
            },
        )
        .vector()
        .get_local()
    )
    isnan_eL0_normalized = np.argwhere(np.isnan(eL0_normalized)).flatten()

    eR0_normalized = (
        df.project(
            eR0_normalized,
            df.VectorFunctionSpace(mesh, "DG", 0),
            form_compiler_parameters={
                "representation": "uflacs",
                "quadrature_degree": deg,
            },
        )
        .vector()
        .get_local()
    )
    isnan_eR0_normalized = np.argwhere(np.isnan(eR0_normalized)).flatten()

    mesh_coordinates = df.FunctionSpace(
        mesh, "DG", 0
    ).tabulate_dof_coordinates()

    np.set_printoptions(threshold=sys.maxsize)
    if isnan_eC0_normalized.size != 0:
        for p in isnan_eC0_normalized:
            list_of_nan_ids = [p // 3 * 3 + i for i in range(0, 3)]
            distances_to_nan_pt = [
                np.linalg.norm(mesh_coordinates[i] - mesh_coordinates[p // 3])
                for i in range(0, len(mesh_coordinates))
            ]
            distances_to_nan_pt[p // 3] = 1000
            closest_pt_id = np.argmin(distances_to_nan_pt, axis=0)
            eC0_normalized[p // 3 * 3 : p // 3 * 3 + 3] = eC0_normalized[
                3 * closest_pt_id : 3 * closest_pt_id + 3
            ]

    if isnan_eL0_normalized.size != 0:
        for p in isnan_eL0_normalized:
            list_of_nan_ids = [p // 3 * 3 + i for i in range(0, 3)]
            distances_to_nan_pt = [
                np.linalg.norm(mesh_coordinates[i] - mesh_coordinates[p // 3])
                for i in range(0, len(mesh_coordinates))
            ]
            distances_to_nan_pt[p // 3] = 1000
            closest_pt_id = np.argmin(distances_to_nan_pt, axis=0)
            eL0_normalized[p // 3 * 3 : p // 3 * 3 + 3] = eL0_normalized[
                3 * closest_pt_id : 3 * closest_pt_id + 3
            ]

    if isnan_eR0_normalized.size != 0:
        for p in isnan_eR0_normalized:
            list_of_nan_ids = [p // 3 * 3 + i for i in range(0, 3)]
            distances_to_nan_pt = [
                np.linalg.norm(mesh_coordinates[i] - mesh_coordinates[p // 3])
                for i in range(0, len(mesh_coordinates))
            ]
            distances_to_nan_pt[p // 3] = 1000
            closest_pt_id = np.argmin(distances_to_nan_pt, axis=0)
            eR0_normalized[p // 3 * 3 : p // 3 * 3 + 3] = eR0_normalized[
                3 * closest_pt_id : 3 * closest_pt_id + 3
            ]

    eC0_normalized_ = df.Function(df.VectorFunctionSpace(mesh, "DG", 0))
    eL0_normalized_ = df.Function(df.VectorFunctionSpace(mesh, "DG", 0))
    eR0_normalized_ = df.Function(df.VectorFunctionSpace(mesh, "DG", 0))

    eC0_normalized_.vector()[:] = eC0_normalized
    eL0_normalized_.vector()[:] = eL0_normalized
    eR0_normalized_.vector()[:] = eR0_normalized

    return eC0_normalized_, eL0_normalized_, eR0_normalized_

def compute_strain(IODet, SimDet, LVid=1, RVid=2, cycle=None):

    # ---------------------------------------------------------------------
    # MPI setup
    # ---------------------------------------------------------------------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ---------------------------------------------------------------------
    # Load mesh + displacement
    # ---------------------------------------------------------------------
    mesh = df.Mesh()
    hdf = df.HDF5File(mesh.mpi_comm(),
                      f"{IODet['outputfolder']}/{IODet['caseID']}/Data.h5",
                      "r")

    u_arr = extractdisplacement(IODet, SimDet, cycle=None)

    # ---------------------------------------------------------------------
    # Load mechanics mesh object (LV or BiV)
    # ---------------------------------------------------------------------
    isLV  = SimDet.get("isLV", False)
    isBiV = SimDet.get("isBiV", False)

    mesh_params = {
        "directory":      IODet["directory_me"],
        "casename":       IODet["casename_me"],
        "casename_marked":IODet["casename_me"],
        "outputfolder":   IODet["outputfolder"],
        "foldername":     IODet["folderName"],
        "isLV":           isLV,
        "isBiV":          isBiV
    }

    if isLV:
        Mesh_obj = lv_mechanics_mesh(mesh_params, SimDet)
    else:
        Mesh_obj = biv_mechanics_mesh(mesh_params, SimDet)

    # ---------------------------------------------------------------------
    # === LOAD PRECOMPUTED eC0, eL0, eR0 FROM BiV_fibers.h5 (Quadrature) ===
    # ---------------------------------------------------------------------
    fiberfile = os.path.join(IODet["directory_me"],"BiV_fibers.h5")
    group = "BiV_fibers"  # group name inside BiV_fibers.h5

    # Quadrature vector space (same as in SetBiVFiber_Quad_PyQ)
    VQuadelem = df.VectorElement("Quadrature",
                                 Mesh_obj.mesh.ufl_cell(),
                                 degree=4,
                                 quad_scheme="default")
    VQuadelem._quad_scheme = "default"
    fiberCLR = df.FunctionSpace(Mesh_obj.mesh, VQuadelem)

    eC0 = df.Function(fiberCLR)
    eL0 = df.Function(fiberCLR)
    eR0 = df.Function(fiberCLR)

    if rank == 0:
        print("📂 Loading fibers from:", fiberfile)
        h5fib = df.HDF5File(Mesh_obj.mesh.mpi_comm(), fiberfile, "r")
        h5fib.read(eC0, f"{group}/eC0")
        h5fib.read(eL0, f"{group}/eL0")
        h5fib.read(eR0, f"{group}/eR0")
        h5fib.close()

    # ---------------------------------------------------------------------
    # Broadcast fiber DOFs to all MPI ranks
    # ---------------------------------------------------------------------
    for vecfun in [eC0, eL0, eR0]:
        if rank == 0:
            gvec = vecfun.vector().get_local()
        else:
            gvec = None
        gvec = comm.bcast(gvec, root=0)
        lo, hi = vecfun.vector().local_range()
        vecfun.vector().set_local(gvec[lo:hi])
        vecfun.vector().apply("insert")

    # ---------------------------------------------------------------------
    # Normalize fiber coordinate system
    # ---------------------------------------------------------------------
    deg = SimDet["GiccioneParams"]["deg"]

    eC0_normalized = eC0 / df.sqrt(df.inner(eC0, eC0))
    eL0_normalized = eL0 / df.sqrt(df.inner(eL0, eL0))
    eR0_normalized = eR0 / df.sqrt(df.inner(eR0, eR0))

    # ---------------------------------------------------------------------
    # Setup displacement field
    # ---------------------------------------------------------------------
    var_deg = 1 if SimDet["Mechanics Discretization"] == "P1P1" else 2
    udisp = df.Function(df.VectorFunctionSpace(Mesh_obj.mesh, "CG", var_deg))

    # ---------------------------------------------------------------------
    # Precompute reference deformation Fref
    # ---------------------------------------------------------------------
    GuccioneParams = SimDet["GiccioneParams"]
    params = {
        "mesh": Mesh_obj.mesh,
        "displacement_variable": udisp,
        "material model": GuccioneParams["Passive model"],
        "material params": GuccioneParams["Passive params"],
        "incompressible": GuccioneParams["incompressible"],
        "growth_tensor": None,
    }

    uflforms = Forms(params)
    Fref = df.project(
        uflforms.Fmat(),
        df.TensorFunctionSpace(Mesh_obj.mesh, "DG", 0),
        form_compiler_parameters={"representation": "uflacs",
                                  "quadrature_degree": deg}
    )

    # ---------------------------------------------------------------------
    # Create outputs
    # ---------------------------------------------------------------------
    Ecc_outdirectory = f"{IODet['outputfolder']}/{IODet['caseID']}/Ecc"
    Ell_outdirectory = f"{IODet['outputfolder']}/{IODet['caseID']}/Ell"
    Err_outdirectory = f"{IODet['outputfolder']}/{IODet['caseID']}/Err"

    for d in [Ecc_outdirectory, Ell_outdirectory, Err_outdirectory]:
        if rank == 0 and not os.path.exists(d):
            os.mkdir(d)
    comm.Barrier()

    File_Ecc = df.File(f"{Ecc_outdirectory}/Ecc.pvd")
    File_Ell = df.File(f"{Ell_outdirectory}/Ell.pvd")
    File_Err = df.File(f"{Err_outdirectory}/Err.pvd")

    Ecc_arr = []
    Ell_arr = []
    Err_arr = []
    Ecc_arr_RV = []
    Ell_arr_RV = []
    Err_arr_RV = []

    # Function space for DG0 fields (declare once)
    FS0 = df.FunctionSpace(Mesh_obj.mesh, "DG", 0)

    # ---------------------------------------------------------------------
    # Loop over time steps
    # ---------------------------------------------------------------------
    for u_arr_ in u_arr:

        # --- MPI broadcast of displacement vector ---
        if rank == 0:
            gvec = u_arr_.vector().get_local()
        else:
            gvec = None
        gvec = comm.bcast(gvec, root=0)
        lo, hi = u_arr_.vector().local_range()
        u_arr_.vector().set_local(gvec[lo:hi])
        u_arr_.vector().apply("insert")

        # update udisp
        udisp.vector()[:] = u_arr_.vector().get_local()

        # Deformation tensors
        Fmat = uflforms.Fmat()
        F = Fmat * df.inv(Fref)
        Cmat = F.T * F

        # Strains
        Ccc = df.inner(eC0_normalized, Cmat * eC0_normalized)
        Cll = df.inner(eL0_normalized, Cmat * eL0_normalized)
        Crr = df.inner(eR0_normalized, Cmat * eR0_normalized)

        Ecc = 0.5 * (1 - 1.0 / Ccc)
        Ell = 0.5 * (1 - 1.0 / Cll)
        Err = 0.5 * (1 - 1.0 / Crr)

        # LV volume
        if isinstance(LVid, int):
            wall_vol = df.assemble(df.Constant(1.0) * Mesh_obj.dx(LVid))
        else:
            wall_vol = 0.0
            for LVid_ in LVid:
                wall_vol += df.assemble(df.Constant(1.0) * Mesh_obj.dx(LVid_))

        # RV volume (only if BiV)
        if isBiV:
            wall_vol_RV = df.assemble(df.Constant(1.0) * Mesh_obj.dx(RVid))

        # --- Integrate global strain (handle LVid int vs list) ---
        if isinstance(LVid, int):
            global_Ecc = df.assemble(Ecc * Mesh_obj.dx(LVid)) / wall_vol
            global_Ell = df.assemble(Ell * Mesh_obj.dx(LVid)) / wall_vol
            global_Err = df.assemble(Err * Mesh_obj.dx(LVid)) / wall_vol
        else:
            global_Ecc = 0.0
            global_Ell = 0.0
            global_Err = 0.0
            for LVid_ in LVid:
                global_Ecc += df.assemble(Ecc * Mesh_obj.dx(LVid_))
                global_Ell += df.assemble(Ell * Mesh_obj.dx(LVid_))
                global_Err += df.assemble(Err * Mesh_obj.dx(LVid_))
            global_Ecc /= wall_vol
            global_Ell /= wall_vol
            global_Err /= wall_vol

        Ecc_arr.append(global_Ecc)
        Ell_arr.append(global_Ell)
        Err_arr.append(global_Err)

        if isBiV:
            Ecc_arr_RV.append(df.assemble(Ecc * Mesh_obj.dx(RVid)) / wall_vol_RV)
            Ell_arr_RV.append(df.assemble(Ell * Mesh_obj.dx(RVid)) / wall_vol_RV)
            Err_arr_RV.append(df.assemble(Err * Mesh_obj.dx(RVid)) / wall_vol_RV)

        # --- Output DG0 strain fields (with explicit quadrature settings) ---
        Ecc_field = df.project(
            Ecc, FS0,
            form_compiler_parameters={
                "representation": "uflacs",
                "quadrature_degree": deg
            }
        )
        Ell_field = df.project(
            Ell, FS0,
            form_compiler_parameters={
                "representation": "uflacs",
                "quadrature_degree": deg
            }
        )
        Err_field = df.project(
            Err, FS0,
            form_compiler_parameters={
                "representation": "uflacs",
                "quadrature_degree": deg
            }
        )

        Ecc_field.rename("Ecc", "Ecc")
        Ell_field.rename("Ell", "Ell")
        Err_field.rename("Err", "Err")

        File_Ecc << Ecc_field
        File_Ell << Ell_field
        File_Err << Err_field

    # ---------------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------------
    if rank == 0:
        print("Maximum Ecc:", min(Ecc_arr))
        print("Maximum Ell:", min(Ell_arr))
        print("Maximum Err:", max(Err_arr))

    # ---------------------------------------------------------------------
    # Save time-series strain arrays
    # ---------------------------------------------------------------------
    if rank == 0:
        np.savez(f"{Ecc_outdirectory}/Ecc.npz", Ecc_arr)
        np.savez(f"{Ell_outdirectory}/Ell.npz", Ell_arr)
        np.savez(f"{Err_outdirectory}/Err.npz", Err_arr)

        if isBiV:
            np.savez(f"{Ecc_outdirectory}/Ecc_RV.npz", Ecc_arr_RV)
            np.savez(f"{Ell_outdirectory}/Ell_RV.npz", Ell_arr_RV)
            np.savez(f"{Err_outdirectory}/Err_RV.npz", Err_arr_RV)


    plt.figure()
    plt.plot(np.arange(0, len(Ecc_arr))*SimDet["writeStep"]*SimDet["dt"], Ecc_arr)
    plt.xlabel("Time point", fontsize=14)
    plt.ylabel("Strain", fontsize=14)
    plt.savefig(os.path.join(Ecc_outdirectory, "Ecc.png"))
    plt.clf()

    plt.figure()
    plt.plot(np.arange(0, len(Ell_arr))*SimDet["writeStep"]*SimDet["dt"], Ell_arr)
    plt.xlabel("Time point", fontsize=14)
    plt.ylabel("Strain", fontsize=14)
    plt.savefig(os.path.join(Ell_outdirectory, "Ell.png"))
    plt.clf()

    plt.figure()
    plt.plot(np.arange(0, len(Err_arr))*SimDet["writeStep"]*SimDet["dt"], Err_arr)
    plt.xlabel("Time point", fontsize=14)
    plt.ylabel("Strain", fontsize=14)
    plt.savefig(os.path.join(Err_outdirectory, "Err.png"))
    plt.clf()

    if isBiV:
        print("Maximum RV Ecc :", min(Ecc_arr_RV))
        print("Maximum RV Ell :", min(Ell_arr_RV))
        print("Maximum RV Err :", max(Err_arr_RV))

        np.savez(os.path.join(Ecc_outdirectory, "Ecc_RV.npz"), Ecc_arr_RV)
        np.savez(os.path.join(Ell_outdirectory, "Ell_RV.npz"), Ell_arr_RV)
        np.savez(os.path.join(Err_outdirectory, "Err_RV.npz"), Err_arr_RV)

        plt.figure()
        plt.plot(np.arange(0, len(Ecc_arr_RV))*SimDet["writeStep"]*SimDet["dt"], Ecc_arr_RV)
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(Ecc_outdirectory, "Ecc_RV.png"))
        plt.clf()

        plt.figure()
        plt.plot(np.arange(0, len(Ell_arr_RV))*SimDet["writeStep"]*SimDet["dt"], Ell_arr_RV)
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(Ell_outdirectory, "Ell_RV.png"))
        plt.clf()

        plt.figure()
        plt.plot(np.arange(0, len(Err_arr_RV))*SimDet["writeStep"]*SimDet["dt"], Err_arr_RV)
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(Err_outdirectory, "Err_RV.png"))
        plt.clf()
    # # ---------------------------------------------------------------------
    # # MPI setup
    # # ---------------------------------------------------------------------
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()

    # # ---------------------------------------------------------------------
    # # Load mesh + displacement
    # # ---------------------------------------------------------------------
    # mesh = df.Mesh()
    # hdf = df.HDF5File(mesh.mpi_comm(),f"{IODet['outputfolder']}/{IODet['caseID']}/Data.h5","r")

    # u_arr = extractdisplacement(IODet, SimDet, cycle=None)

    # # ---------------------------------------------------------------------
    # # Load mechanics mesh object (LV or BiV)
    # # ---------------------------------------------------------------------
    # isLV  = SimDet.get("isLV", False)
    # isBiV = SimDet.get("isBiV", False)

    # mesh_params = {
    #     "directory": IODet["directory_me"],
    #     "casename": IODet["casename_me"],
    #     "casename_marked": IODet["casename_me"],
    #     "outputfolder": IODet["outputfolder"],
    #     "foldername": IODet["folderName"],
    #     "isLV": isLV,
    #     "isBiV": isBiV
    # }

    # if isLV:
    #     Mesh_obj = lv_mechanics_mesh(mesh_params, SimDet)
    # else:
    #     Mesh_obj = biv_mechanics_mesh(mesh_params, SimDet)

    # # ---------------------------------------------------------------------
    # # === NEW: LOAD PRECOMPUTED eC0, eL0, eR0 FIELDS FROM HDF5 ===
    # # ---------------------------------------------------------------------

    # fiberfile = os.path.join(IODet["directory_me"],"refined11_pj-baseline-update/BiV_fibers.h5",)
    # casename  = IODet["casename_me"]
    # # In the file the root group is 'BiV_fibers', not casename
    # group = "BiV_fibers"

    # VQuadelem = df.VectorElement("Quadrature", Mesh_obj.mesh.ufl_cell(), degree=4, quad_scheme="default")
    # VQuadelem._quad_scheme = "default"

    # fiberCLR = df.FunctionSpace(Mesh_obj.mesh, VQuadelem)

    # eC0 = df.Function(fiberCLR)
    # eL0 = df.Function(fiberCLR)
    # eR0 = df.Function(fiberCLR)

    # if rank == 0:
    #     print("📂 Loading fibers from:", fiberfile)
    #     h5fib = df.HDF5File(Mesh_obj.mesh.mpi_comm(), fiberfile, "r")
    #     h5fib.read(eC0, f"{group}/eC0")
    #     h5fib.read(eL0, f"{group}/eL0")
    #     h5fib.read(eR0, f"{group}/eR0")
    #     h5fib.close()


    # # ---------------------------------------------------------------------
    # # Broadcast fiber DOFs to all MPI ranks
    # # ---------------------------------------------------------------------
    # for vecfun in [eC0, eL0, eR0]:
    #     if rank == 0:
    #         gvec = vecfun.vector().get_local()
    #     else:
    #         gvec = None

    #     # send entire vector to all ranks
    #     gvec = comm.bcast(gvec, root=0)

    #     lo, hi = vecfun.vector().local_range()
    #     vecfun.vector().set_local(gvec[lo:hi])
    #     vecfun.vector().apply("insert")

    # # ---------------------------------------------------------------------
    # # Normalize fiber coordinate system
    # # ---------------------------------------------------------------------
    # deg = SimDet["GiccioneParams"]["deg"]

    # eC0_normalized = eC0 / df.sqrt(df.inner(eC0, eC0))
    # eL0_normalized = eL0 / df.sqrt(df.inner(eL0, eL0))
    # eR0_normalized = eR0 / df.sqrt(df.inner(eR0, eR0))

    # # ---------------------------------------------------------------------
    # # Setup displacement field
    # # ---------------------------------------------------------------------
    # var_deg = 1 if SimDet["Mechanics Discretization"] == "P1P1" else 2
    # udisp = df.Function(df.VectorFunctionSpace(Mesh_obj.mesh, "CG", var_deg))

    # # ---------------------------------------------------------------------
    # # Precompute reference deformation
    # # ---------------------------------------------------------------------
    # GuccioneParams = SimDet["GiccioneParams"]
    # params = {
    #     "mesh": Mesh_obj.mesh,
    #     "displacement_variable": udisp,
    #     "material model": GuccioneParams["Passive model"],
    #     "material params": GuccioneParams["Passive params"],
    #     "incompressible": GuccioneParams["incompressible"],
    #     "growth_tensor": None,
    # }

    # uflforms = Forms(params)
    # Fref = df.project(uflforms.Fmat(),
    #                   df.TensorFunctionSpace(Mesh_obj.mesh, "DG", 0))

    # # ---------------------------------------------------------------------
    # # Create outputs
    # # ---------------------------------------------------------------------
    # Ecc_outdirectory = f"{IODet['outputfolder']}/{IODet['caseID']}/Ecc"
    # Ell_outdirectory = f"{IODet['outputfolder']}/{IODet['caseID']}/Ell"
    # Err_outdirectory = f"{IODet['outputfolder']}/{IODet['caseID']}/Err"

    # for d in [Ecc_outdirectory, Ell_outdirectory, Err_outdirectory]:
    #     if rank == 0 and not os.path.exists(d):
    #         os.mkdir(d)
    # comm.Barrier()

    # File_Ecc = df.File(f"{Ecc_outdirectory}/Ecc.pvd")
    # File_Ell = df.File(f"{Ell_outdirectory}/Ell.pvd")
    # File_Err = df.File(f"{Err_outdirectory}/Err.pvd")

    # Ecc_arr = []
    # Ell_arr = []
    # Err_arr = []
    # Ecc_arr_RV = []
    # Ell_arr_RV = []
    # Err_arr_RV = []

    # # ---------------------------------------------------------------------
    # # Loop over time steps
    # # ---------------------------------------------------------------------
    # for u_arr_ in u_arr:

    #     # --- MPI broadcast of displacement vector ---
    #     if rank == 0:
    #         gvec = u_arr_.vector().get_local()
    #     else:
    #         gvec = None

    #     gvec = comm.bcast(gvec, root=0)

    #     lo, hi = u_arr_.vector().local_range()
    #     u_arr_.vector().set_local(gvec[lo:hi])
    #     u_arr_.vector().apply("insert")

    #     # update udisp
    #     udisp.vector()[:] = u_arr_.vector().get_local()

    #     # Deformation tensors
    #     Fmat = uflforms.Fmat()
    #     F = Fmat * df.inv(Fref)
    #     Cmat = F.T * F

    #     # Strains
    #     Ccc = df.inner(eC0_normalized, Cmat * eC0_normalized)
    #     Cll = df.inner(eL0_normalized, Cmat * eL0_normalized)
    #     Crr = df.inner(eR0_normalized, Cmat * eR0_normalized)

    #     Ecc = 0.5 * (1 - 1 / Ccc)
    #     Ell = 0.5 * (1 - 1 / Cll)
    #     Err = 0.5 * (1 - 1 / Crr)

    #     # LV volume
    #     if isinstance(LVid, int):
    #         wall_vol = df.assemble(df.Constant(1.0) * Mesh_obj.dx(LVid))
    #     else:
    #         wall_vol = sum(df.assemble(df.Constant(1.0) * Mesh_obj.dx(LVid_))
    #                        for LVid_ in LVid)

    #     if isBiV:
    #         wall_vol_RV = df.assemble(df.Constant(1.0) * Mesh_obj.dx(RVid))


    #     # --- Integrate global strain over LV ---
    #     if isinstance(LVid, int):
    #         global_Ecc = df.assemble(Ecc * Mesh_obj.dx(LVid)) / wall_vol
    #         global_Ell = df.assemble(Ell * Mesh_obj.dx(LVid)) / wall_vol
    #         global_Err = df.assemble(Err * Mesh_obj.dx(LVid)) / wall_vol
    #     else:
    #         global_Ecc = 0.0
    #         global_Ell = 0.0
    #         global_Err = 0.0
    #         for LVid_ in LVid:
    #             global_Ecc += df.assemble(Ecc * Mesh_obj.dx(LVid_))
    #             global_Ell += df.assemble(Ell * Mesh_obj.dx(LVid_))
    #             global_Err += df.assemble(Err * Mesh_obj.dx(LVid_))
    #         global_Ecc /= wall_vol
    #         global_Ell /= wall_vol
    #         global_Err /= wall_vol

    #     Ecc_arr.append(global_Ecc)
    #     Ell_arr.append(global_Ell)
    #     Err_arr.append(global_Err)

    #     if isBiV:
    #         Ecc_arr_RV.append(df.assemble(Ecc * Mesh_obj.dx(RVid)) / wall_vol_RV)
    #         Ell_arr_RV.append(df.assemble(Ell * Mesh_obj.dx(RVid)) / wall_vol_RV)
    #         Err_arr_RV.append(df.assemble(Err * Mesh_obj.dx(RVid)) / wall_vol_RV)

    #     # # --- Integrate global strain
    #     # global_Ecc = df.assemble(Ecc * Mesh_obj.dx(LVid)) / wall_vol
    #     # global_Ell = df.assemble(Ell * Mesh_obj.dx(LVid)) / wall_vol
    #     # global_Err = df.assemble(Err * Mesh_obj.dx(LVid)) / wall_vol

    #     # Ecc_arr.append(global_Ecc)
    #     # Ell_arr.append(global_Ell)
    #     # Err_arr.append(global_Err)

    #     # if isBiV:
    #     #     Ecc_arr_RV.append(df.assemble(Ecc * Mesh_obj.dx(RVid)) / wall_vol_RV)
    #     #     Ell_arr_RV.append(df.assemble(Ell * Mesh_obj.dx(RVid)) / wall_vol_RV)
    #     #     Err_arr_RV.append(df.assemble(Err * Mesh_obj.dx(RVid)) / wall_vol_RV)

    #     # --- Output DG0 strain fields
    #     FS0 = df.FunctionSpace(Mesh_obj.mesh, "DG", 0)

    #     Ecc_field = df.project(Ecc, FS0)
    #     Ell_field = df.project(Ell, FS0)
    #     Err_field = df.project(Err, FS0)

    #     Ecc_field.rename("Ecc", "Ecc")
    #     Ell_field.rename("Ell", "Ell")
    #     Err_field.rename("Err", "Err")

    #     File_Ecc << Ecc_field
    #     File_Ell << Ell_field
    #     File_Err << Err_field

    # # ---------------------------------------------------------------------
    # # Print results
    # # ---------------------------------------------------------------------
    # if rank == 0:
    #     print("Maximum Ecc:", min(Ecc_arr))
    #     print("Maximum Ell:", min(Ell_arr))
    #     print("Maximum Err:", max(Err_arr))

    # # ---------------------------------------------------------------------
    # # Save time-series strain arrays
    # # ---------------------------------------------------------------------
    # if rank == 0:
    #     np.savez(f"{Ecc_outdirectory}/Ecc.npz", Ecc_arr)
    #     np.savez(f"{Ell_outdirectory}/Ell.npz", Ell_arr)
    #     np.savez(f"{Err_outdirectory}/Err.npz", Err_arr)

    #     if isBiV:
    #         np.savez(f"{Ecc_outdirectory}/Ecc_RV.npz", Ecc_arr_RV)
    #         np.savez(f"{Ell_outdirectory}/Ell_RV.npz", Ell_arr_RV)
    #         np.savez(f"{Err_outdirectory}/Err_RV.npz", Err_arr_RV)


# def compute_strain(IODet, SimDet, LVid=1, RVid=2, cycle=None):
#     isLV = False
#     isBiV = False

#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()

#     mesh = df.Mesh()
#     hdf = df.HDF5File(mesh.mpi_comm(),IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5","r",)

#     u_arr = extractdisplacement(IODet, SimDet, cycle=None)

#     if "isLV" in list(SimDet.keys()):
#         isLV = SimDet["isLV"]

#         if(isLV):
#             mesh_params = {
#                 "directory": IODet["directory_me"],
#                 "casename": IODet["casename_me"],
#                 "outputfolder": IODet["outputfolder"],
#                 "foldername": IODet["folderName"],
#                 "isLV": isLV,
#             }

#             Mesh_obj = lv_mechanics_mesh(mesh_params, SimDet)
#             try:
#                 eC0 = Mesh_obj.eC0
#                 eL0 = Mesh_obj.eL0
#                 eR0 = Mesh_obj.eR0

#             # Mesh_obj = lv_mechanics_mesh(mesh_params, SimDet)

#             # try:
#             #     eC0 = Mesh_obj.eC0
#             #     eL0 = Mesh_obj.eL0
#             #     eR0 = Mesh_obj.eR0

#             except AttributeError:

#                 fiber_angle_param = {
#                     "mesh": Mesh_obj.mesh,
#                     "facetboundaries": Mesh_obj.facetboundaries,
#                     "LV_fiber_angle": [0.01, -0.01],
#                     "LV_sheet_angle": [0.1, -0.1],
#                     "minztol": Mesh_obj.mesh.hmax()/2.0, # Coarse mesh
#                     "isrotatept": False,
#                     "isreturn": True,
#                     "outfilename": IODet["casename_me"],
#                     "outdirectory": IODet["outputfolder"] + IODet["caseID"] + IODet["folderName"],
#                     "baseid": SimDet["topid"],
#                     "epiid": SimDet["epiid"],
#                     "lvid": SimDet["LVendoid"],
#                     "degree": SimDet["GiccioneParams"]["deg"]
#                 }

#                 eC0, eL0, eR0  = vtk_py.addLVfiber_LDRB(fiber_angle_param)
#                 eC0, eL0, eR0  = vtk_py.addLVfiber_LDRB(fiber_angle_param)

#     # if "isBiV" in list(SimDet.keys()):
#     #     isBiV = SimDet["isBiV"]

#     #     if(isBiV):
#     #         mesh_params = {
#     #             "directory": IODet["directory_me"],
#     #             "casename": IODet["casename_me"],
#     #             "casename_marked": IODet["casename_me"],                
#     #             "outputfolder": IODet["outputfolder"],
#     #             "foldername": IODet["folderName"],
#     #             "isBiV": isBiV,
#     #         }

#     #         Mesh_obj = biv_mechanics_mesh(mesh_params, SimDet)


#     #         try:
#     #             eC0 = Mesh_obj.eC0
#     #             eL0 = Mesh_obj.eL0
#     #             eR0 = Mesh_obj.eR0

#     #         except AttributeError:

#     #             # Set BiVFiber
#     #             baseid = [SimDet["topid"]]
#     #             if "RVtopid" in SimDet.keys():
#     #                 baseid.append(SimDet["RVtopid"])
#     #             if "LVtopid" in SimDet.keys():
#     #                 baseid.append(SimDet["LVtopid"])

#     #             fiber_angle_param = {"mesh": Mesh_obj.mesh,\
#     #             	 "facetboundaries": Mesh_obj.facetboundaries,\
#     #             	 "LV_fiber_angle": [0.01,-0.01], \
#     #             	 "LV_sheet_angle": [0.1, -0.1], \
#     #             	 "Septum_fiber_angle": [0.01, -0.01],\
#     #             	 "Septum_sheet_angle": [0.1, -0.1],\
#     #             	 "RV_fiber_angle": [0.01, -0.01],\
#     #             	 "RV_sheet_angle": [0.1, -0.1],\
#     #             	 "LV_matid": 0,\
#     #             	 "Septum_matid": 1,\
#     #             	 "RV_matid": 2,\
#     #             	 "matid":  Mesh_obj.matid,\
#     #             	 "isrotatept": False,\
#     #             	 "isreturn": True,\
#     #                  "outfilename": IODet["casename_me"],
#     #                  "outdirectory": IODet["outputfolder"] + IODet["caseID"] + IODet["folderName"],
#     #                  "baseid": baseid,
#     #             	 "epiid": SimDet["epiid"],\
#     #             	 "rvid": SimDet["RVendoid"],\
#     #             	 "lvid": SimDet["LVendoid"],\
#     #             	 "degree": 4}

#     #             eC0, eL0, eR0 = vtk_py.SetBiVFiber_Quad_PyQ(fiber_angle_param)

#     # MPI - SAFE
#     # --- initialize so we don't get UnboundLocalError on ranks ---
#     eC0 = None
#     eL0 = None
#     eR0 = None

#     if "isBiV" in list(SimDet.keys()):

#         isBiV = SimDet["isBiV"]

#         if(isBiV):
#             mesh_params = {
#                 "directory": IODet["directory_me"],
#                 "casename": IODet["casename_me"],
#                 "casename_marked": IODet["casename_me"],                
#                 "outputfolder": IODet["outputfolder"],
#                 "foldername": IODet["folderName"],
#                 "isBiV": isBiV,
#             }

#             Mesh_obj = biv_mechanics_mesh(mesh_params, SimDet)

#             # --- Fiber computation only on rank 0 ---
#             if rank == 0:
#                 try:
#                     eC0 = Mesh_obj.eC0
#                     eL0 = Mesh_obj.eL0
#                     eR0 = Mesh_obj.eR0

#                 except AttributeError:
#                     print("[Rank 0] Computing fibers using PyQ...")

#                     # Set BiVFiber
#                     baseid = [SimDet["topid"]]
#                     if "RVtopid" in SimDet.keys():
#                         baseid.append(SimDet["RVtopid"])
#                     if "LVtopid" in SimDet.keys():
#                         baseid.append(SimDet["LVtopid"])

#                     fiber_angle_param = {"mesh": Mesh_obj.mesh,\
#                         "facetboundaries": Mesh_obj.facetboundaries,\
#                         "LV_fiber_angle": [0.01,-0.01], \
#                         "LV_sheet_angle": [0.1, -0.1], \
#                         "Septum_fiber_angle": [0.01, -0.01],\
#                         "Septum_sheet_angle": [0.1, -0.1],\
#                         "RV_fiber_angle": [0.01, -0.01],\
#                         "RV_sheet_angle": [0.1, -0.1],\
#                         "LV_matid": 0,\
#                         "Septum_matid": 1,\
#                         "RV_matid": 2,\
#                         "matid":  Mesh_obj.matid,\
#                         "isrotatept": False,\
#                         "isreturn": True,\
#                         "outfilename": IODet["casename_me"],
#                         "outdirectory": IODet["outputfolder"] + IODet["caseID"] + IODet["folderName"],
#                         "baseid": baseid,
#                         "epiid": SimDet["epiid"],\
#                         "rvid": SimDet["RVendoid"],\
#                         "lvid": SimDet["LVendoid"],\
#                         "degree": 4}
                
#                     eC0, eL0, eR0 = vtk_py.SetBiVFiber_Quad_PyQ(fiber_angle_param)


#     # --- DOF-correct broadcast of fiber fields ---
#     for vec in [eC0, eL0, eR0]:
#         if rank == 0:
#             gvec = vec.vector().get_local()
#         else:
#             gvec = None

#         gvec = comm.bcast(gvec, root=0)
#         lo, hi = vec.vector().local_range()
#         vec.vector().set_local(gvec[lo:hi])
#         vec.vector().apply("insert")
#         # # 1. Rank 0 has the global coefficients
#         # if rank == 0:
#         #     gvec = vec.vector().get_local()
#         # else:
#         #     gvec = None

#         # # 2. Broadcast full global vector to all ranks
#         # gvec = comm.bcast(gvec, root=0)

#         # # 3. Each rank extracts its local slice of DOFs
#         # lo, hi = vec.vector().local_range()
#         # lvec = gvec[lo:hi]

#         # # 4. Write into the local FEniCS vector
#         # vec.vector().set_local(lvec)
#         # vec.vector().apply("insert")

#     # --- Broadcast segmentation markers ---
#     for mf in [Mesh_obj.AHAid, Mesh_obj.matid, Mesh_obj.baseid]:
        
#         if rank == 0:
#             arr = mf.array()
#         else:
#             arr = None

#         arr = comm.bcast(arr, root=0)
#         mf.set_values(arr)        

#     # deg = SimDet["GiccioneParams"]["deg"]
#     #eC0_normalized, eL0_normalized, eR0_normalized = normalize_directionalbasis(
#     #    eC0, eL0, eR0, Mesh_obj.mesh, deg
#     #)
#     deg = SimDet["GiccioneParams"]["deg"]

#     eC0_normalized = eC0 / df.sqrt(df.inner(eC0, eC0))
#     eL0_normalized = eL0 / df.sqrt(df.inner(eL0, eL0))
#     eR0_normalized = eR0 / df.sqrt(df.inner(eR0, eR0))


#     if SimDet["Mechanics Discretization"] is "P1P1":
#         var_deg = 1
#     else:
#         var_deg = 2

#     udisp = df.Function(df.VectorFunctionSpace(Mesh_obj.mesh, "CG", var_deg))
#     udisp.vector()[:] = u_arr[0].vector().get_local()[:]

#     GuccioneParams = SimDet["GiccioneParams"]
#     params = {
#         "mesh": Mesh_obj.mesh,
#         "displacement_variable": udisp,
#         "material model": GuccioneParams["Passive model"],
#         "material params": GuccioneParams["Passive params"],
#         "incompressible": GuccioneParams["incompressible"],
#         "growth_tensor": None,
#     }

#     uflforms = Forms(params)
#     Fref = df.project(uflforms.Fmat(), df.TensorFunctionSpace(Mesh_obj.mesh, "DG", 0))

#     Ecc_outdirectory = os.path.join(IODet["outputfolder"], IODet["caseID"], "Ecc")
#     if not os.path.exists(Ecc_outdirectory):
#         os.mkdir(Ecc_outdirectory)
#     #df.File(os.path.join(Ecc_outdirectory, "Ecc_direction.pvd")) << eC0_normalized
#     File_Ecc = df.File(os.path.join(Ecc_outdirectory, "Ecc.pvd"))
#     Ecc_arr = []
#     Ecc_arr_RV = []

#     Ell_outdirectory = os.path.join(IODet["outputfolder"], IODet["caseID"], "Ell")
#     if not os.path.exists(Ell_outdirectory):
#         os.mkdir(Ell_outdirectory)
#     #df.File(os.path.join(Ell_outdirectory, "Ell_direction.pvd")) << eL0_normalized
#     File_Ell = df.File(os.path.join(Ell_outdirectory, "Ell.pvd"))
#     Ell_arr = []
#     Ell_arr_RV = []

#     Err_outdirectory = os.path.join(IODet["outputfolder"], IODet["caseID"], "Err")
#     if not os.path.exists(Err_outdirectory):
#         os.mkdir(Err_outdirectory)
#     #df.File(os.path.join(Err_outdirectory, "Err_direction.pvd")) << eR0_normalized
#     File_Err = df.File(os.path.join(Err_outdirectory, "Err.pvd"))
#     Err_arr = []
#     Err_arr_RV = []

#     for u_arr_ in u_arr:
#         # --- DOF-correct broadcast of displacement field u_arr_ ---
#         if rank == 0:
#             gvec = u_arr_.vector().get_local()
#         else:
#             gvec = None

#         # Broadcast the full global displacement vector to all ranks
#         gvec = comm.bcast(gvec, root=0)

#         # Each rank extracts its own DOF slice
#         lo, hi = u_arr_.vector().local_range()
#         lvec = gvec[lo:hi]

#         u_arr_.vector().set_local(lvec)
#         u_arr_.vector().apply("insert")
#         # -----------------------------------------------------------

#         if isinstance(LVid, int):
#             wall_vol = df.assemble(df.Constant(1.0) * Mesh_obj.dx(LVid),form_compiler_parameters={"representation": "uflacs"})
#         elif isinstance(LVid, list):
#             wall_vol = 0
#             for LVid_ in LVid:
#                 wall_vol += df.assemble(df.Constant(1.0) * Mesh_obj.dx(LVid_),form_compiler_parameters={"representation": "uflacs"})        

#         # if isinstance(LVid, int):
#         #     wall_vol = df.assemble(df.Constant(1.0) * Mesh_obj.dx(LVid),form_compiler_parameters={"representation": "uflacs"},)
#         # elif isinstance(LVid, list):
#         #     wall_vol = 0
#         #     for LVid_ in LVid:
#         #         wall_vol += df.assemble(df.Constant(1.0) * Mesh_obj.dx(LVid_),form_compiler_parameters={"representation": "uflacs"},)

#         wall_vol_RV = df.assemble(df.Constant(1.0) * Mesh_obj.dx(RVid),form_compiler_parameters={"representation": "uflacs"},)

#         # --- DOF-correct broadcast of displacement field u_arr_ ---
#         if rank == 0:
#             gvec = u_arr_.vector().get_local()
#         else:
#             gvec = None

#         gvec = comm.bcast(gvec, root=0)

#         # each rank extracts its local slice
#         lo, hi = u_arr_.vector().local_range()
#         u_arr_.vector().set_local(gvec[lo:hi])
#         u_arr_.vector().apply("insert")

#         udisp.vector()[:] = u_arr_.vector().get_local()[:]

#         Fmat = uflforms.Fmat()
#         F = Fmat * df.inv(Fref)
#         Cmat = F.T * F

#         Ccc = df.inner(eC0_normalized, Cmat * eC0_normalized)
#         Ecc = 0.5 * (1 - 1 / Ccc)

#         if isinstance(LVid, int):
#             global_Ecc = (df.assemble(Ecc * Mesh_obj.dx(LVid),form_compiler_parameters={"representation": "uflacs"},)/ wall_vol)
#         elif isinstance(LVid, list):
#             global_Ecc = 0
#             for LVid_ in LVid:
#                 global_Ecc += (df.assemble(Ecc * Mesh_obj.dx(LVid_),form_compiler_parameters={"representation": "uflacs"},))
#             global_Ecc = global_Ecc/wall_vol 

#         Ecc_arr.append(global_Ecc)
#         if isBiV:
#             global_Ecc_RV = (df.assemble(Ecc * Mesh_obj.dx(RVid),form_compiler_parameters={"representation": "uflacs"},)/ wall_vol)
#             Ecc_arr_RV.append(global_Ecc_RV)

#         Ecc_field = df.project(Ecc,df.FunctionSpace(Mesh_obj.mesh, "DG", 0),form_compiler_parameters={"representation": "uflacs","quadrature_degree": deg,},)
#         Ecc_field.rename("Ecc", "Ecc")
#         File_Ecc << Ecc_field
#         Ecc_arr.append(global_Ecc)

#         Cll = df.inner(eL0_normalized, Cmat * eL0_normalized)
#         Ell = 0.5 * (1 - 1 / Cll)
#         if isinstance(LVid, int):
#             global_Ell = (df.assemble(Ell * Mesh_obj.dx(LVid),form_compiler_parameters={"representation": "uflacs"},)/ wall_vol)
#         elif isinstance(LVid, list):
#             global_Ell = 0
#             for LVid_ in LVid:
#                 global_Ell += (df.assemble(Ell * Mesh_obj.dx(LVid_),form_compiler_parameters={"representation": "uflacs"},))
#             global_Ell = global_Ell/wall_vol 

#         Ell_arr.append(global_Ell)
#         if isBiV:
#             global_Ell_RV = (df.assemble(Ell * Mesh_obj.dx(RVid),form_compiler_parameters={"representation": "uflacs"},)/ wall_vol)
#             Ell_arr_RV.append(global_Ell_RV)

#         Ell_field = df.project(Ell,df.FunctionSpace(Mesh_obj.mesh, "DG", 0),form_compiler_parameters={"representation": "uflacs","quadrature_degree": deg,},)
#         Ell_field.rename("Ell", "Ell")
#         File_Ell << Ell_field
#         Ell_arr.append(global_Ell)

#         Crr = df.inner(eR0_normalized, Cmat * eR0_normalized)
#         Err = 0.5 * (1 - 1 / Crr)
#         if isinstance(LVid, int):
#             global_Err = (df.assemble(Err * Mesh_obj.dx(LVid),form_compiler_parameters={"representation": "uflacs"},)/ wall_vol)
#         elif isinstance(LVid, list):
#             global_Err = 0
#             for LVid_ in LVid:
#                 global_Err += (df.assemble(Err * Mesh_obj.dx(LVid_),form_compiler_parameters={"representation": "uflacs"},))
#             global_Err = global_Err/wall_vol 

#         Err_arr.append(global_Err)
#         if isBiV:
#             global_Err_RV = (df.assemble(Err * Mesh_obj.dx(RVid),form_compiler_parameters={"representation": "uflacs"},)/ wall_vol)
#             Err_arr_RV.append(global_Err_RV)

#         Err_field = df.project(Err,df.FunctionSpace(Mesh_obj.mesh, "DG", 0),form_compiler_parameters={"representation": "uflacs","quadrature_degree": deg,},)
#         Err_field.rename("Err", "Err")
#         File_Err << Err_field
#         Err_arr.append(global_Err)

#     print("Maximum Ecc :", min(Ecc_arr))
#     print("Maximum Ell :", min(Ell_arr))
#     print("Maximum Err :", max(Err_arr))

#     np.savez(os.path.join(Ecc_outdirectory, "Ecc.npz"), Ecc_arr)
#     np.savez(os.path.join(Ell_outdirectory, "Ell.npz"), Ell_arr)
#     np.savez(os.path.join(Err_outdirectory, "Err.npz"), Err_arr)

#     plt.figure()
#     plt.plot(np.arange(0, len(Ecc_arr)), Ecc_arr)
#     plt.xlabel("Time point", fontsize=14)
#     plt.ylabel("Strain", fontsize=14)
#     plt.savefig(os.path.join(Ecc_outdirectory, "Ecc.png"))
#     plt.clf()

#     plt.figure()
#     plt.plot(np.arange(0, len(Ell_arr)), Ell_arr)
#     plt.xlabel("Time point", fontsize=14)
#     plt.ylabel("Strain", fontsize=14)
#     plt.savefig(os.path.join(Ell_outdirectory, "Ell.png"))
#     plt.clf()

#     plt.figure()
#     plt.plot(np.arange(0, len(Err_arr)), Err_arr)
#     plt.xlabel("Time point", fontsize=14)
#     plt.ylabel("Strain", fontsize=14)
#     plt.savefig(os.path.join(Err_outdirectory, "Err.png"))
#     plt.clf()

#     if isBiV:
#         print("Maximum RV Ecc :", min(Ecc_arr_RV))
#         print("Maximum RV Ell :", min(Ell_arr_RV))
#         print("Maximum RV Err :", max(Err_arr_RV))

#         np.savez(os.path.join(Ecc_outdirectory, "Ecc_RV.npz"), Ecc_arr_RV)
#         np.savez(os.path.join(Ell_outdirectory, "Ell_RV.npz"), Ell_arr_RV)
#         np.savez(os.path.join(Err_outdirectory, "Err_RV.npz"), Err_arr_RV)

#         plt.figure()
#         plt.plot(np.arange(0, len(Ecc_arr_RV)), Ecc_arr_RV)
#         plt.xlabel("Time point", fontsize=14)
#         plt.ylabel("Strain", fontsize=14)
#         plt.savefig(os.path.join(Ecc_outdirectory, "Ecc_RV.png"))
#         plt.clf()

#         plt.figure()
#         plt.plot(np.arange(0, len(Ell_arr_RV)), Ell_arr_RV)
#         plt.xlabel("Time point", fontsize=14)
#         plt.ylabel("Strain", fontsize=14)
#         plt.savefig(os.path.join(Ell_outdirectory, "Ell_RV.png"))
#         plt.clf()

#         plt.figure()
#         plt.plot(np.arange(0, len(Err_arr_RV)), Err_arr_RV)
#         plt.xlabel("Time point", fontsize=14)
#         plt.ylabel("Strain", fontsize=14)
#         plt.savefig(os.path.join(Err_outdirectory, "Err_RV.png"))
#         plt.clf()


def compute_strain_split(IODet, SimDet, LVid=1, RVid=2, cycle=None, axis_split=None):

    mesh = df.Mesh()
    hdf = df.HDF5File(mesh.mpi_comm(),IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5","r",)
    u_arr = extractdisplacement(IODet, SimDet, cycle=None)

    if "isLV" in list(SimDet.keys()):
        isLV = SimDet["isLV"]
        if(isLV):
            mesh_params = { "directory": IODet["directory_me"],
                            "casename": IODet["casename_me"],
                            "outputfolder": IODet["outputfolder"],
                            "foldername": IODet["folderName"],
                            "isLV": isLV,}

            Mesh_obj = lv_mechanics_mesh(mesh_params, SimDet)
            try:
                eC0 = Mesh_obj.eC0
                eL0 = Mesh_obj.eL0
                eR0 = Mesh_obj.eR0

            except AttributeError:

                fiber_angle_param = {   "mesh": Mesh_obj.mesh,
                                        "facetboundaries": Mesh_obj.facetboundaries,
                                        "LV_fiber_angle": [0.01, -0.01],
                                        "LV_sheet_angle": [0.1, -0.1],
                                        "minztol": Mesh_obj.mesh.hmax()/2.0, # Coarse mesh
                                        "isrotatept": False,
                                        "isreturn": True,
                                        "outfilename": IODet["casename_me"],
                                        "outdirectory": IODet["outputfolder"] + IODet["caseID"] + IODet["folderName"],
                                        "baseid": SimDet["topid"],
                                        "epiid": SimDet["epiid"],
                                        "lvid": SimDet["LVendoid"],
                                        "degree": SimDet["GiccioneParams"]["deg"]}

                eC0, eL0, eR0  = vtk_py.addLVfiber_LDRB(fiber_angle_param)

    deg = SimDet["GiccioneParams"]["deg"]
    eC0_normalized = eC0 / df.sqrt(df.inner(eC0, eC0))
    eL0_normalized = eL0 / df.sqrt(df.inner(eL0, eL0))
    eR0_normalized = eR0 / df.sqrt(df.inner(eR0, eR0))

    if SimDet["Mechanics Discretization"] is "P1P1":
        var_deg = 1
    else:
        var_deg = 2

    udisp = df.Function(df.VectorFunctionSpace(Mesh_obj.mesh, "CG", var_deg))
    udisp.vector()[:] = u_arr[0].vector().get_local()[:]

    GuccioneParams = SimDet["GiccioneParams"]
    params = {  "mesh": Mesh_obj.mesh,
                "displacement_variable": udisp,
                "material model": GuccioneParams["Passive model"],
                "material params": GuccioneParams["Passive params"],
                "incompressible": GuccioneParams["incompressible"],
                "growth_tensor": None,}

    uflforms = Forms(params)
    Fref = df.project(uflforms.Fmat(), df.TensorFunctionSpace(Mesh_obj.mesh, "DG", 0))

    Ecc_outdirectory = os.path.join(IODet["outputfolder"], IODet["caseID"], "Ecc")
    if not os.path.exists(Ecc_outdirectory):
        os.mkdir(Ecc_outdirectory)

    File_Ecc = df.File(os.path.join(Ecc_outdirectory, "Ecc.pvd"))
    Ecc_arr = []
    Ecc_arr_RV = []

    Ell_outdirectory = os.path.join(IODet["outputfolder"], IODet["caseID"], "Ell")
    if not os.path.exists(Ell_outdirectory):
        os.mkdir(Ell_outdirectory)

    File_Ell = df.File(os.path.join(Ell_outdirectory, "Ell.pvd"))
    Ell_arr = []
    Ell_arr_RV = []

    Err_outdirectory = os.path.join(IODet["outputfolder"], IODet["caseID"], "Err")
    if not os.path.exists(Err_outdirectory):
        os.mkdir(Err_outdirectory)

    File_Err = df.File(os.path.join(Err_outdirectory, "Err.pvd"))
    Err_arr = []
    Err_arr_RV = []

    if axis_split is not None:

        left_cells = df.MeshFunction("size_t", Mesh_obj.mesh, Mesh_obj.mesh.topology().dim())
        left_cells.set_all(0)
        axis, split_val = axis_split
        for cell in df.cells(Mesh_obj.mesh):
            coord = cell.midpoint()     
            if getattr(coord, axis)() <= split_val:
                left_cells[cell.index()] = 1 #left
            else:
                left_cells[cell.index()] = 2 #right

        dx_lr = df.Measure("dx", domain=Mesh_obj.mesh, subdomain_data=left_cells)
        wall_vol_left = df.assemble(df.Constant(1.0) * dx_lr(1))
        wall_vol_right = df.assemble(df.Constant(1.0) * dx_lr(2))        
        Ecc_left, Ecc_right = [], []
        Ell_left, Ell_right = [], []
        Err_left, Err_right = [], []    

    for u_arr_ in u_arr:
        # --- STEP 4: DOF-CORRECT BROADCAST OF THE DISPLACEMENT FIELD u_arr_ ---
        if rank == 0:
            gvec = u_arr_.vector().get_local()
        else:
            gvec = None

        # Broadcast global displacement vector
        gvec = comm.bcast(gvec, root=0)

        # Local DOF range (SAFE for MPI!)
        lo, hi = u_arr_.vector().local_range()
        lvec = gvec[lo:hi]

        # Put correct slice into local vector
        u_arr_.vector().set_local(lvec)
        u_arr_.vector().apply("insert")
        # --------------------------------------------------------------

        if isinstance(LVid, int):
            wall_vol = df.assemble(df.Constant(1.0) * Mesh_obj.dx(LVid),form_compiler_parameters={"representation": "uflacs"},)
        elif isinstance(LVid, list):
            wall_vol = 0
            for LVid_ in LVid:
                wall_vol += df.assemble(df.Constant(1.0) * Mesh_obj.dx(LVid_),form_compiler_parameters={"representation": "uflacs"},)

        udisp.vector()[:] = u_arr_.vector().get_local()[:]

        Fmat = uflforms.Fmat()
        F = Fmat * df.inv(Fref)
        Cmat = F.T * F

        # Circumferential strain
        Ccc = df.inner(eC0_normalized, Cmat * eC0_normalized)
        Ecc = 0.5 * (1 - 1 / Ccc)
        if isinstance(LVid, int):
            global_Ecc = (df.assemble(Ecc * Mesh_obj.dx(LVid),form_compiler_parameters={"representation": "uflacs"},)/ wall_vol)
        elif isinstance(LVid, list):
            global_Ecc = 0
            for LVid_ in LVid:
                global_Ecc += (df.assemble(Ecc * Mesh_obj.dx(LVid_),form_compiler_parameters={"representation": "uflacs"},))
            global_Ecc = global_Ecc/wall_vol 

        # Ecc_arr.append(global_Ecc)
        Ecc_field = df.project(Ecc,df.FunctionSpace(Mesh_obj.mesh, "DG", 0),form_compiler_parameters={"representation": "uflacs","quadrature_degree": deg,},)
        Ecc_field.rename("Ecc", "Ecc")
        File_Ecc << Ecc_field
        Ecc_arr.append(global_Ecc)

        # Longitudinal strain
        Cll = df.inner(eL0_normalized, Cmat * eL0_normalized)
        Ell = 0.5 * (1 - 1 / Cll)
        if isinstance(LVid, int):
            global_Ell = (df.assemble(Ell * Mesh_obj.dx(LVid),form_compiler_parameters={"representation": "uflacs"},)/ wall_vol)
        elif isinstance(LVid, list):
            global_Ell = 0
            for LVid_ in LVid:
                global_Ell += (df.assemble(Ell * Mesh_obj.dx(LVid_),form_compiler_parameters={"representation": "uflacs"},))
            global_Ell = global_Ell/wall_vol 

        # Ell_arr.append(global_Ell)
        Ell_field = df.project(Ell,df.FunctionSpace(Mesh_obj.mesh, "DG", 0),form_compiler_parameters={"representation": "uflacs","quadrature_degree": deg,},)
        Ell_field.rename("Ell", "Ell")
        File_Ell << Ell_field
        Ell_arr.append(global_Ell)

        # Radial strain 
        Crr = df.inner(eR0_normalized, Cmat * eR0_normalized)
        Err = 0.5 * (1 - 1 / Crr)
        if isinstance(LVid, int):
            global_Err = (df.assemble(Err * Mesh_obj.dx(LVid),form_compiler_parameters={"representation": "uflacs"},)/ wall_vol)
        elif isinstance(LVid, list):
            global_Err = 0
            for LVid_ in LVid:
                global_Err += (df.assemble(Err * Mesh_obj.dx(LVid_),form_compiler_parameters={"representation": "uflacs"},))
            global_Err = global_Err/wall_vol 

        # Err_arr.append(global_Err)        
        Err_field = df.project(Err,df.FunctionSpace(Mesh_obj.mesh, "DG", 0),form_compiler_parameters={"representation": "uflacs","quadrature_degree": deg,},)
        Err_field.rename("Err", "Err")
        File_Err << Err_field
        Err_arr.append(global_Err)


        

        # Hemisphere-specific averages
        if axis_split is not None:
            left_avg_Ecc = df.assemble(Ecc * dx_lr(1)) / wall_vol_left
            right_avg_Ecc = df.assemble(Ecc * dx_lr(2)) / wall_vol_right
            Ecc_left.append(left_avg_Ecc)
            Ecc_right.append(right_avg_Ecc)

            left_avg_Ell = df.assemble(Ell * dx_lr(1)) / wall_vol_left
            right_avg_Ell = df.assemble(Ell * dx_lr(2)) / wall_vol_right
            Ell_left.append(left_avg_Ell)
            Ell_right.append(right_avg_Ell)

            left_avg_Err = df.assemble(Err * dx_lr(1)) / wall_vol_left
            right_avg_Err = df.assemble(Err * dx_lr(2)) / wall_vol_right
            Err_left.append(left_avg_Err)
            Err_right.append(right_avg_Err)        

    print("=== Global Strain Summary ===")
    print("Maximum Ecc :", min(Ecc_arr))
    print("Maximum Ell :", min(Ell_arr))
    print("Maximum Err :", max(Err_arr))

    if axis_split is not None:
            print("=== Left Hemisphere Max Strains ===")
            print("Maximum Ecc:", min(Ecc_left))
            print("Maximum Ell:", min(Ell_left))
            print("Maximum Err:", max(Err_left))

            print("=== Right Hemisphere Max Strains ===")
            print("Maximum Ecc:", min(Ecc_right))
            print("Maximum Ell:", min(Ell_right))
            print("Maximum Err:", max(Err_right))    

    np.savez(os.path.join(Ecc_outdirectory, "Ecc.npz"), Ecc_arr)
    np.savez(os.path.join(Ell_outdirectory, "Ell.npz"), Ell_arr)
    np.savez(os.path.join(Err_outdirectory, "Err.npz"), Err_arr)

    if axis_split is not None:
        np.savez(os.path.join(Ecc_outdirectory, "Ecc_hemispheres.npz"), left=Ecc_left, right=Ecc_right)
        np.savez(os.path.join(Ell_outdirectory, "Ell_hemispheres.npz"), left=Ell_left, right=Ell_right)
        np.savez(os.path.join(Err_outdirectory, "Err_hemispheres.npz"), left=Err_left, right=Err_right)


    # Plot global strains
    plt.figure()
    plt.plot(np.arange(0, len(Ecc_arr)), Ecc_arr)
    plt.xlabel("Time point", fontsize=14)
    plt.ylabel("Strain", fontsize=14)
    plt.savefig(os.path.join(Ecc_outdirectory, "Ecc.png"))
    plt.clf()

    plt.figure()
    plt.plot(np.arange(0, len(Ell_arr)), Ell_arr)
    plt.xlabel("Time point", fontsize=14)
    plt.ylabel("Strain", fontsize=14)
    plt.savefig(os.path.join(Ell_outdirectory, "Ell.png"))
    plt.clf()

    plt.figure()
    plt.plot(np.arange(0, len(Err_arr)), Err_arr)
    plt.xlabel("Time point", fontsize=14)
    plt.ylabel("Strain", fontsize=14)
    plt.savefig(os.path.join(Err_outdirectory, "Err.png"))
    plt.clf()

    if axis_split is not None:
        plt.figure()
        plt.plot(np.arange(0, len(Ecc_left)), Ecc_left, label="Left")
        plt.plot(np.arange(0, len(Ecc_right)), Ecc_right, label="Right")
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(Ecc_outdirectory, "Ecc_hemispheres.png"))
        plt.clf()

        plt.figure()
        plt.plot(np.arange(0, len(Ell_left)), Ell_left, label="Left")
        plt.plot(np.arange(0, len(Ell_right)), Ell_right, label="Right")
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(Ell_outdirectory, "Ell_hemispheres.png"))
        plt.clf()

        plt.figure()
        plt.plot(np.arange(0, len(Err_left)), Err_left, label="Left")
        plt.plot(np.arange(0, len(Err_right)), Err_right, label="Right")
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(Err_outdirectory, "Err_hemispheres.png"))
        plt.clf()

    if isBiV:
        print("Maximum RV Ecc :", min(Ecc_arr_RV))
        print("Maximum RV Ell :", min(Ell_arr_RV))
        print("Maximum RV Err :", max(Err_arr_RV))

        np.savez(os.path.join(Ecc_outdirectory, "Ecc_RV.npz"), Ecc_arr_RV)
        np.savez(os.path.join(Ell_outdirectory, "Ell_RV.npz"), Ell_arr_RV)
        np.savez(os.path.join(Err_outdirectory, "Err_RV.npz"), Err_arr_RV)

        plt.figure()
        plt.plot(np.arange(0, len(Ecc_arr_RV)), Ecc_arr_RV)
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(Ecc_outdirectory, "Ecc_RV.png"))
        plt.clf()

        plt.figure()
        plt.plot(np.arange(0, len(Ell_arr_RV)), Ell_arr_RV)
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(Ell_outdirectory, "Ell_RV.png"))
        plt.clf()

        plt.figure()
        plt.plot(np.arange(0, len(Err_arr_RV)), Err_arr_RV)
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(Err_outdirectory, "Err_RV.png"))
        plt.clf()

def compute_strain_split(IODet, SimDet, LVid=1, RVid=2, cycle=None, axis_split=None):

    mesh = df.Mesh()
    hdf = df.HDF5File(mesh.mpi_comm(),IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5","r",)
    u_arr = extractdisplacement(IODet, SimDet, cycle=None)

    if "isLV" in list(SimDet.keys()):
        isLV = SimDet["isLV"]
        if(isLV):
            mesh_params = { "directory": IODet["directory_me"],
                            "casename": IODet["casename_me"],
                            "outputfolder": IODet["outputfolder"],
                            "foldername": IODet["folderName"],
                            "isLV": isLV,}

            Mesh_obj = lv_mechanics_mesh(mesh_params, SimDet)
            try:
                eC0 = Mesh_obj.eC0
                eL0 = Mesh_obj.eL0
                eR0 = Mesh_obj.eR0

            except AttributeError:

                fiber_angle_param = {   "mesh": Mesh_obj.mesh,
                                        "facetboundaries": Mesh_obj.facetboundaries,
                                        "LV_fiber_angle": [0.01, -0.01],
                                        "LV_sheet_angle": [0.1, -0.1],
                                        "minztol": Mesh_obj.mesh.hmax()/2.0, # Coarse mesh
                                        "isrotatept": False,
                                        "isreturn": True,
                                        "outfilename": IODet["casename_me"],
                                        "outdirectory": IODet["outputfolder"] + IODet["caseID"] + IODet["folderName"],
                                        "baseid": SimDet["topid"],
                                        "epiid": SimDet["epiid"],
                                        "lvid": SimDet["LVendoid"],
                                        "degree": SimDet["GiccioneParams"]["deg"]}

                eC0, eL0, eR0  = vtk_py.addLVfiber_LDRB(fiber_angle_param)

    deg = SimDet["GiccioneParams"]["deg"]
    eC0_normalized = eC0 / df.sqrt(df.inner(eC0, eC0))
    eL0_normalized = eL0 / df.sqrt(df.inner(eL0, eL0))
    # eR0_normalized = eR0 / df.sqrt(df.inner(eR0, eR0))

    if SimDet["Mechanics Discretization"] is "P1P1":
        var_deg = 1
    else:
        var_deg = 2

    udisp = df.Function(df.VectorFunctionSpace(Mesh_obj.mesh, "CG", var_deg))
    udisp.vector()[:] = u_arr[0].vector().get_local()[:]

    GuccioneParams = SimDet["GiccioneParams"]
    params = {  "mesh": Mesh_obj.mesh,
                "displacement_variable": udisp,
                "material model": GuccioneParams["Passive model"],
                "material params": GuccioneParams["Passive params"],
                "incompressible": GuccioneParams["incompressible"],
                "growth_tensor": None,}

    uflforms = Forms(params)
    Fref = df.project(uflforms.Fmat(), df.TensorFunctionSpace(Mesh_obj.mesh, "DG", 0))

    Ecc_outdirectory = os.path.join(IODet["outputfolder"], IODet["caseID"], "Ecc")
    if not os.path.exists(Ecc_outdirectory):
        os.mkdir(Ecc_outdirectory)

    File_Ecc = df.File(os.path.join(Ecc_outdirectory, "Ecc.pvd"))
    Ecc_arr = []
    Ecc_arr_RV = []

    Ell_outdirectory = os.path.join(IODet["outputfolder"], IODet["caseID"], "Ell")
    if not os.path.exists(Ell_outdirectory):
        os.mkdir(Ell_outdirectory)

    File_Ell = df.File(os.path.join(Ell_outdirectory, "Ell.pvd"))
    Ell_arr = []
    Ell_arr_RV = []

    # Err_outdirectory = os.path.join(IODet["outputfolder"], IODet["caseID"], "Err")
    # if not os.path.exists(Err_outdirectory):
    #     os.mkdir(Err_outdirectory)

    # File_Err = df.File(os.path.join(Err_outdirectory, "Err.pvd"))
    # Err_arr = []
    # Err_arr_RV = []

    if axis_split is not None:

        left_cells = df.MeshFunction("size_t", Mesh_obj.mesh, Mesh_obj.mesh.topology().dim())
        left_cells.set_all(0)
        axis, split_val = axis_split
        for cell in df.cells(Mesh_obj.mesh):
            coord = cell.midpoint()     
            if getattr(coord, axis)() <= split_val:
                left_cells[cell.index()] = 1 #left
            else:
                left_cells[cell.index()] = 2 #right

        dx_lr = df.Measure("dx", domain=Mesh_obj.mesh, subdomain_data=left_cells)
        wall_vol_left = df.assemble(df.Constant(1.0) * dx_lr(1))
        wall_vol_right = df.assemble(df.Constant(1.0) * dx_lr(2))        
        Ecc_left, Ecc_right = [], []
        Ell_left, Ell_right = [], []
        # Err_left, Err_right = [], []    

    for u_arr_ in u_arr:

        if isinstance(LVid, int):
            wall_vol = df.assemble(df.Constant(1.0) * Mesh_obj.dx(LVid),form_compiler_parameters={"representation": "uflacs"},)
        elif isinstance(LVid, list):
            wall_vol = 0
            for LVid_ in LVid:
                wall_vol += df.assemble(df.Constant(1.0) * Mesh_obj.dx(LVid_),form_compiler_parameters={"representation": "uflacs"},)

        udisp.vector()[:] = u_arr_.vector().get_local()[:]

        Fmat = uflforms.Fmat()
        F = Fmat * df.inv(Fref)
        Cmat = F.T * F

        # Circumferential strain
        Ccc = df.inner(eC0_normalized, Cmat * eC0_normalized)
        Ecc = 0.5 * (1 - 1 / Ccc)
        if isinstance(LVid, int):
            global_Ecc = (df.assemble(Ecc * Mesh_obj.dx(LVid),form_compiler_parameters={"representation": "uflacs"},)/ wall_vol)
        elif isinstance(LVid, list):
            global_Ecc = 0
            for LVid_ in LVid:
                global_Ecc += (df.assemble(Ecc * Mesh_obj.dx(LVid_),form_compiler_parameters={"representation": "uflacs"},))
            global_Ecc = global_Ecc/wall_vol 

        # Ecc_arr.append(global_Ecc)
        Ecc_field = df.project(Ecc,df.FunctionSpace(Mesh_obj.mesh, "DG", 0),form_compiler_parameters={"representation": "uflacs","quadrature_degree": deg,},)
        Ecc_field.rename("Ecc", "Ecc")
        File_Ecc << Ecc_field
        Ecc_arr.append(global_Ecc)

        # Longitudinal strain
        Cll = df.inner(eL0_normalized, Cmat * eL0_normalized)
        Ell = 0.5 * (1 - 1 / Cll)
        if isinstance(LVid, int):
            global_Ell = (df.assemble(Ell * Mesh_obj.dx(LVid),form_compiler_parameters={"representation": "uflacs"},)/ wall_vol)
        elif isinstance(LVid, list):
            global_Ell = 0
            for LVid_ in LVid:
                global_Ell += (df.assemble(Ell * Mesh_obj.dx(LVid_),form_compiler_parameters={"representation": "uflacs"},))
            global_Ell = global_Ell/wall_vol 

        # Ell_arr.append(global_Ell)
        Ell_field = df.project(Ell,df.FunctionSpace(Mesh_obj.mesh, "DG", 0),form_compiler_parameters={"representation": "uflacs","quadrature_degree": deg,},)
        Ell_field.rename("Ell", "Ell")
        File_Ell << Ell_field
        Ell_arr.append(global_Ell)

        # # Radial strain 
        # Crr = df.inner(eR0_normalized, Cmat * eR0_normalized)
        # Err = 0.5 * (1 - 1 / Crr)
        # if isinstance(LVid, int):
        #     global_Err = (df.assemble(Err * Mesh_obj.dx(LVid),form_compiler_parameters={"representation": "uflacs"},)/ wall_vol)
        # elif isinstance(LVid, list):
        #     global_Err = 0
        #     for LVid_ in LVid:
        #         global_Err += (df.assemble(Err * Mesh_obj.dx(LVid_),form_compiler_parameters={"representation": "uflacs"},))
        #     global_Err = global_Err/wall_vol 

        # # Err_arr.append(global_Err)        
        # Err_field = df.project(Err,df.FunctionSpace(Mesh_obj.mesh, "DG", 0),form_compiler_parameters={"representation": "uflacs","quadrature_degree": deg,},)
        # Err_field.rename("Err", "Err")
        # File_Err << Err_field
        # Err_arr.append(global_Err)


        

        # Hemisphere-specific averages
        if axis_split is not None:
            left_avg_Ecc = df.assemble(Ecc * dx_lr(1)) / wall_vol_left
            right_avg_Ecc = df.assemble(Ecc * dx_lr(2)) / wall_vol_right
            Ecc_left.append(left_avg_Ecc)
            Ecc_right.append(right_avg_Ecc)

            left_avg_Ell = df.assemble(Ell * dx_lr(1)) / wall_vol_left
            right_avg_Ell = df.assemble(Ell * dx_lr(2)) / wall_vol_right
            Ell_left.append(left_avg_Ell)
            Ell_right.append(right_avg_Ell)

            # left_avg_Err = df.assemble(Err * dx_lr(1)) / wall_vol_left
            # right_avg_Err = df.assemble(Err * dx_lr(2)) / wall_vol_right
            # Err_left.append(left_avg_Err)
            # Err_right.append(right_avg_Err)        

    print("=== Global Strain Summary ===")
    print("Maximum Ecc :", min(Ecc_arr))
    print("Maximum Ell :", min(Ell_arr))
    # print("Maximum Err :", max(Err_arr))

    if axis_split is not None:
            print("=== Left Hemisphere Max Strains ===")
            print("Maximum Ecc:", min(Ecc_left))
            print("Maximum Ell:", min(Ell_left))
            # print("Maximum Err:", max(Err_left))

            print("=== Right Hemisphere Max Strains ===")
            print("Maximum Ecc:", min(Ecc_right))
            print("Maximum Ell:", min(Ell_right))
            # print("Maximum Err:", max(Err_right))    

    np.savez(os.path.join(Ecc_outdirectory, "Ecc.npz"), Ecc_arr)
    np.savez(os.path.join(Ell_outdirectory, "Ell.npz"), Ell_arr)
    # np.savez(os.path.join(Err_outdirectory, "Err.npz"), Err_arr)

    if axis_split is not None:
        np.savez(os.path.join(Ecc_outdirectory, "Ecc_hemispheres.npz"), left=Ecc_left, right=Ecc_right)
        np.savez(os.path.join(Ell_outdirectory, "Ell_hemispheres.npz"), left=Ell_left, right=Ell_right)
        # np.savez(os.path.join(Err_outdirectory, "Err_hemispheres.npz"), left=Err_left, right=Err_right)


    # Plot global strains
    plt.figure()
    plt.plot(np.arange(0, len(Ecc_arr)), Ecc_arr)
    plt.xlabel("Time point", fontsize=14)
    plt.ylabel("Strain", fontsize=14)
    plt.savefig(os.path.join(Ecc_outdirectory, "Ecc.png"))
    plt.clf()

    plt.figure()
    plt.plot(np.arange(0, len(Ell_arr)), Ell_arr)
    plt.xlabel("Time point", fontsize=14)
    plt.ylabel("Strain", fontsize=14)
    plt.savefig(os.path.join(Ell_outdirectory, "Ell.png"))
    plt.clf()

    # plt.figure()
    # plt.plot(np.arange(0, len(Err_arr)), Err_arr)
    # plt.xlabel("Time point", fontsize=14)
    # plt.ylabel("Strain", fontsize=14)
    # plt.savefig(os.path.join(Err_outdirectory, "Err.png"))
    # plt.clf()

    if axis_split is not None:
        plt.figure()
        plt.plot(np.arange(0, len(Ecc_left)), Ecc_left, label="Left")
        plt.plot(np.arange(0, len(Ecc_right)), Ecc_right, label="Right")
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(Ecc_outdirectory, "Ecc_hemispheres.png"))
        plt.clf()

        plt.figure()
        plt.plot(np.arange(0, len(Ell_left)), Ell_left, label="Left")
        plt.plot(np.arange(0, len(Ell_right)), Ell_right, label="Right")
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(Ell_outdirectory, "Ell_hemispheres.png"))
        plt.clf()

        # plt.figure()
        # plt.plot(np.arange(0, len(Err_left)), Err_left, label="Left")
        # plt.plot(np.arange(0, len(Err_right)), Err_right, label="Right")
        # plt.xlabel("Time point", fontsize=14)
        # plt.ylabel("Strain", fontsize=14)
        # plt.legend()
        # plt.savefig(os.path.join(Err_outdirectory, "Err_hemispheres.png"))
        # plt.clf()


def extractdisplacement(IODet, SimDet, cycle=None):

    mesh = df.Mesh()
    hdf = df.HDF5File(
        mesh.mpi_comm(),
        IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5",
        "r",
    )

    # Dump displacement
    if SimDet["Mechanics Discretization"] is "P1P1":
        var_deg = 1
    else:
        var_deg = 2

    try:
        u_arr = extractvtk(
            IODet["outputfolder"] + "/" + IODet["caseID"],
            "ME/" + "u",
            "CG",
            var_deg,
            IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "ME_" + "u",
            "u",
        )
    except RuntimeError:
        print("No attribute for ", var, " found")

    return u_arr


def extractdisplacementloading(IODet, SimDet, cycle=None):

    mesh = df.Mesh()
    hdf = df.HDF5File(
        mesh.mpi_comm(),
        IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5",
        "r",
    )

    # Dump displacement
    if SimDet["Mechanics Discretization"] is "P1P1":
        var_deg = 1
    else:
        var_deg = 2

    try:
        u_arr = extractvtk(
            IODet["outputfolder"] + "/" + IODet["caseID"],
            "ME/" + "u_loading",
            "CG",
            var_deg,
            IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "ME_" + "u_loading",
            "u",
        )
    except RuntimeError:
        print("No attribute for ", var, " found")

    return u_arr


def dumpvtk(IODet, SimDet, cycle=None, ME_var = [], EP_var = [], PJ_var = []):

    hdf = df.HDF5File(df.MPI.comm_world, IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5", "r",)

    list_of_ME_var = ME_var
    list_of_EP_var = EP_var
    list_of_PJ_var = PJ_var

    #list_of_ME_var = [
    #    ["u", "CG", 1],
    #    ["potential_ref", "CG", 1],
    #    ["Ecc", "DG", 0],
    #    ["Ell", "DG", 0],
    #    ["Err", "DG", 0],
    #    ["Eff", "DG", 0],
    #    ["fstress", "DG", 0],
    #    ["imp", "DG", 1],
    #    ["imp2", "DG", 1],
    #    ["imp_constraint", "DG", 1],
    #]

    #list_of_EP_var = [["phi", "CG", 1], ["r", "DG", 0], ["potential_ref", "CG", 1]]
    #list_of_PJ_var = [["phi", "CG", 1], ["r", "DG", 0], ["potential_ref", "CG", 1]]

    if hdf.has_dataset("ME"):
        for ME_var in list_of_ME_var:
            # pdb.set_trace()

            var = ME_var[0]
            var_space = ME_var[1]
            var_deg = ME_var[2]

            print("Extracting ME", var, " ", var_space, " ", var_deg)
            gnenericvar = "field"
            try:
                var_arr = extractvtk(
                    IODet["outputfolder"] + "/" + IODet["caseID"],
                    "ME/" + var,
                    var_space,
                    var_deg,
                    IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "ME_" + var,
                    gnenericvar,
                    group="ME",
                )
            except RuntimeError:
                print("No attribute for ", var, " found")

            if var == "u":
                u_arr = var_arr.copy()

    if hdf.has_dataset("EP"):
        # pdb.set_trace()
        for EP_var in list_of_EP_var:

            var = EP_var[0]
            var_space = EP_var[1]
            var_deg = EP_var[2]
            print("Extracting EP", var)

            try:
                var_arr = extractvtk(IODet["outputfolder"] + "/" + IODet["caseID"],"EP/" + var,var_space,var_deg,IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "EP_" + var,var,group="EP",)

            except RuntimeError:
                print("No attribute for ", var, " found")

    if hdf.has_dataset("PJ"):
        for PJ_var in list_of_PJ_var:

            var = PJ_var[0]
            var_space = PJ_var[1]
            var_deg = PJ_var[2]
            print("Extracting PJ", var)

            try:
                var_arr = extractvtk(IODet["outputfolder"] + "/" + IODet["caseID"],"PJ/" + var,var_space,var_deg,IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "PJ_" + var,var,group="PJ",)

            except RuntimeError:
                print("No attribute for ", var, " found")


def plothemodynamics(IODet, SimDet, cycle=None):

    directory = IODet["outputfolder"] + "/"
    casename = IODet["caseID"]
    BCL = SimDet["HeartBeatLength"]
    plt.figure()
    if cycle is None:
        cycle = SimDet["closedloopparam"]["stop_iter"] + 1
        t_systole= np.zeros(cycle)

        for ncycle in range(cycle):
            filename = directory + casename + "/" + "BiV_PV.txt"
            homo_tptt, homo_LVP, homo_LVV, homo_RVP, homo_RVV, homo_Qmv = extract_PV(
                filename, BCL, ncycle, SimDet
            )
            plt.plot(homo_LVV, homo_LVP, label=f"LV Cycle = {ncycle}")
            if SimDet.get("isBiV") or SimDet.get("isFCH"):
                plt.plot(homo_RVV, homo_RVP, label=f"RV Cycle = {ncycle}")

    else:
        filename = directory + casename + "/" + "BiV_PV.txt"
        homo_tptt, homo_LVP, homo_LVV, homo_RVP, homo_RVV, homo_Qmv = extract_PV(
            filename, BCL, int(cycle), SimDet
        )
        plt.plot(homo_LVV, homo_LVP, label=f"LV Cycle = {cycle}")
        if SimDet.get("isBiV") or SimDet.get("isFCH"):
            plt.plot(homo_RVV, homo_RVP, label=f"RV Cycle = {cycle}")


    hemodynamics_outdirectory = os.path.join(
        IODet["outputfolder"], IODet["caseID"], "hemodynamics"
    )
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    # plt.plot(homo_LVV, homo_LVP)
    plt.savefig(os.path.join(hemodynamics_outdirectory, "PV.png"))
    plt.clf()


def plotpressure(IODet, SimDet, cycle=None, compartment="All"):

    directory = IODet["outputfolder"] + "/"
    casename = IODet["caseID"]
    BCL = SimDet["HeartBeatLength"]
    plt.figure()
    filename = directory + casename + "/" + "BiV_P.txt"

    if cycle is None:
        cycle = SimDet["closedloopparam"]["stop_iter"] + 1

        for ncycle in range(cycle):

            tpt, Psv, PLV, Psa, PLA, Ppv, PRV, Ppa, PRA = extract_P(
                filename, BCL, ncycle, SimDet
            )

            if ncycle == cycle - 1:
                meanPsv = time_average(tpt, Psv)
                print(
                    "Mean Psv = ",
                    meanPsv,
                    "mmHg; Peak Psv = ",
                    max(Psv),
                    "mmHg",
                )
                meanPLV = time_average(tpt, PLV)
                print(
                    "Mean PLV = ",
                    meanPLV,
                    "mmHg; Peak PLV = ",
                    max(PLV),
                    "mmHg",
                )
                meanPsa = time_average(tpt, Psa)
                print(
                    "Mean Psa = ",
                    meanPsa,
                    "mmHg; Peak Psa = ",
                    max(Psa),
                    "mmHg",
                )
                meanPLA = time_average(tpt, PLA)
                print(
                    "Mean PLA = ",
                    meanPLA,
                    "mmHg; Peak PLA = ",
                    max(PLA),
                    "mmHg",
                )

            if compartment == "All" or "sv" in compartment:
                plt.plot(tpt, Psv, label=f"Psv Cycle = {ncycle}")
            if compartment == "All" or "lv" in compartment:
                plt.plot(tpt, PLV, label=f"PLV Cycle = {ncycle}")
            if compartment == "All" or "sa" in compartment:
                plt.plot(tpt, Psa, label=f"Psa Cycle = {ncycle}")
            if compartment == "All" or "la" in compartment:
                plt.plot(tpt, PLA, label=f"PLA Cycle = {ncycle}")

            if "isBiV" in list(SimDet.keys()):
                if SimDet["isBiV"]:
                    if compartment == "All" or "pv" in compartment:
                        plt.plot(tpt, Ppv, label=f"Ppv Cycle = {ncycle}")
                    if compartment == "All" or "rv" in compartment:
                        plt.plot(tpt, PRV, label=f"PRV Cycle = {ncycle}")
                    if compartment == "All" or "pa" in compartment:
                        plt.plot(tpt, Ppa, label=f"Ppa Cycle = {ncycle}")
                    if compartment == "All" or "ra" in compartment:
                        plt.plot(tpt, PRA, label=f"PRA Cycle = {ncycle}")

                    if ncycle == cycle - 1:
                        meanPpv = time_average(tpt, Ppv)
                        print(
                            "Mean Ppv = ",
                            meanPpv,
                            "mmHg; Peak Ppv = ",
                            max(Ppv),
                            "mmHg",
                        )
                        meanPRV = time_average(tpt, PRV)
                        print(
                            "Mean PRV = ",
                            meanPRV,
                            "mmHg; Peak PRV = ",
                            max(PRV),
                            "mmHg",
                        )
                        meanPpa = time_average(tpt, Ppa)
                        print(
                            "Mean Ppa = ",
                            meanPpa,
                            "mmHg; Peak Ppa = ",
                            max(Ppa),
                            "mmHg",
                        )
                        meanPRA = time_average(tpt, PRA)
                        print(
                            "Mean PRA = ",
                            meanPRA,
                            "mmHg; Peak PRA = ",
                            max(PRA),
                            "mmHg",
                        )

    else:
        tpt, Psv, PLV, Psa, PLA, Ppv, PRV, Ppa, PRA = extract_P(
            filename, BCL, int(cycle), SimDet
        )

        meanPsv = time_average(tpt, Psv)
        print(
            "Mean Psv = ",
            meanPsv,
            "mmHg; Peak Psv = ",
            max(Psv),
            "mmHg",
        )
        meanPLV = time_average(tpt, PLV)
        print(
            "Mean PLV = ",
            meanPLV,
            "mmHg; Peak PLV = ",
            max(PLV),
            "mmHg",
        )
        meanPsa = time_average(tpt, Psa)
        print(
            "Mean Psa = ",
            meanPsa,
            "mmHg; Peak Psa = ",
            max(Psa),
            "mmHg",
        )
        meanPLA = time_average(tpt, PLA)
        print(
            "Mean PLA = ",
            meanPLA,
            "mmHg; Peak PLA = ",
            max(PLA),
            "mmHg",
        )

        if compartment == "All" or "sv" in compartment:
            plt.plot(tpt, Psv, label=f"Psv Cycle = {cycle}")
        if compartment == "All" or "lv" in compartment:
            plt.plot(tpt, PLV, label=f"PLV Cycle = {cycle}")
        if compartment == "All" or "sa" in compartment:
            plt.plot(tpt, Psa, label=f"Psa Cycle = {cycle}")
        if compartment == "All" or "la" in compartment:
            plt.plot(tpt, PLA, label=f"PLA Cycle = {cycle}")

        if "isBiV" in list(SimDet.keys()):
            if SimDet["isBiV"]:
                if compartment == "All" or "pv" in compartment:
                    plt.plot(tpt, Ppv, label=f"Ppv Cycle = {cycle}")
                if compartment == "All" or "rv" in compartment:
                    plt.plot(tpt, PRV, label=f"PRV Cycle = {cycle}")
                if compartment == "All" or "pa" in compartment:
                    plt.plot(tpt, Ppa, label=f"Ppa Cycle = {cycle}")
                if compartment == "All" or "ra" in compartment:
                    plt.plot(tpt, PRA, label=f"PRA Cycle = {cycle}")

                meanPpv = time_average(tpt, Ppv)
                print(
                    "Mean Ppv = ",
                    meanPpv,
                    "mmHg; Peak Ppv = ",
                    max(Ppv),
                    "mmHg",
                )
                meanPRV = time_average(tpt, PRV)
                print(
                    "Mean PRV = ",
                    meanPRV,
                    "mmHg; Peak PRV = ",
                    max(PRV),
                    "mmHg",
                )
                meanPpa = time_average(tpt, Ppa)
                print(
                    "Mean Ppa = ",
                    meanPpa,
                    "mmHg; Peak Ppa = ",
                    max(Ppa),
                    "mmHg",
                )
                meanPRA = time_average(tpt, PRA)
                print(
                    "Mean PRA = ",
                    meanPRA,
                    "mmHg; Peak PRA = ",
                    max(PRA),
                    "mmHg",
                )

    hemodynamics_outdirectory = os.path.join(
        IODet["outputfolder"], IODet["caseID"], "hemodynamics"
    )
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    plt.legend()
    plt.ylabel("Pressure (mmHg)")
    plt.xlabel("Time (s)")
    plt.savefig(os.path.join(hemodynamics_outdirectory, "Pressure.png"))
    plt.clf()


def plotflow(IODet, SimDet, cycle=None, compartment="All"):

    directory = IODet["outputfolder"] + "/"
    casename = IODet["caseID"]
    BCL = SimDet["HeartBeatLength"]
    if cycle is None:
        cycle = SimDet["closedloopparam"]["stop_iter"] + 1

    plt.figure()
    for ncycle in range(cycle):
        filename = directory + casename + "/" + "BiV_Q.txt"

        tpt, Qav, Qmv, Qsa, Qsv, Qpvv, Qtv, Qpa, Qpv, Qlvad = extract_Q(
            filename, BCL, ncycle, SimDet
        )

        if ncycle == cycle - 1:
            meanQav = time_average(tpt, Qav)
            print(
                "Mean Qav = ",
                meanQav * 60,
                "L/min; Peak Qav = ",
                max(Qav) * 60,
                "L/min",
            )
            meanQmv = time_average(tpt, Qmv)
            print(
                "Mean Qmv = ",
                meanQmv * 60,
                "L/min; Peak Qmv = ",
                max(Qmv) * 60,
                "L/min",
            )
            meanQsa = time_average(tpt, Qsa)
            print(
                "Mean Qsa = ",
                meanQsa * 60,
                "L/min; Peak Qsa = ",
                max(Qsa) * 60,
                "L/min",
            )
            meanQsv = time_average(tpt, Qsv)
            print(
                "Mean Qsv = ",
                meanQsv * 60,
                "L/min; Peak Qsv = ",
                max(Qsv) * 60,
                "L/min",
            )
            meanQlvad = time_average(tpt, Qlvad)
            print(
                "Mean Qlvad = ",
                meanQlvad * 60,
                "L/min; Peak Qlvad = ",
                max(Qlvad) * 60,
                "L/min",
            )

        if compartment == "All" or "av" in compartment:
            plt.plot(tpt, Qav * 60, label=f"Qav Cycle = {ncycle}")
        if compartment == "All" or "mv" in compartment:
            plt.plot(tpt, Qmv * 60, label=f"Qmv Cycle = {ncycle}")
        if compartment == "All" or "sa" in compartment:
            plt.plot(tpt, Qsa * 60, label=f"Qsa Cycle = {ncycle}")
        if compartment == "All" or "sv" in compartment:
            plt.plot(tpt, Qsv * 60, label=f"Qsv Cycle = {ncycle}")
        if compartment == "All" or "lvad" in compartment:
            plt.plot(tpt, Qlvad * 60, label=f"Qlvad Cycle = {ncycle}")

        if "isBiV" in list(SimDet.keys()):
            if SimDet["isBiV"]:
                if compartment == "All" or "ppv" in compartment:
                    plt.plot(tpt, Qpvv * 60, label=f"Qpvv Cycle = {ncycle}")
                if compartment == "All" or "tv" in compartment:
                    plt.plot(tpt, Qtv * 60, label=f"Qtv Cycle = {ncycle}")
                if compartment == "All" or "pa" in compartment:
                    plt.plot(tpt, Qpa * 60, label=f"Qpa Cycle = {ncycle}")
                if compartment == "All" or "pv" in compartment:
                    plt.plot(tpt, Qpv * 60, label=f"Qpv Cycle = {ncycle}")

                if ncycle == cycle - 1:
                    meanQpvv = time_average(tpt, Qpvv)
                    print(
                        "Mean Qpvv = ",
                        meanQpvv * 60,
                        "L/min; Peak Qpvv = ",
                        max(Qpvv) * 60,
                        "L/min",
                    )
                    meanQtv = time_average(tpt, Qtv)
                    print(
                        "Mean Qtv = ",
                        meanQtv * 60,
                        "L/min; Peak Qtv = ",
                        max(Qtv) * 60,
                        "L/min",
                    )
                    meanQpa = time_average(tpt, Qpa)
                    print(
                        "Mean Qpa = ",
                        meanQpa * 60,
                        "L/min; Peak Qpa = ",
                        max(Qpa) * 60,
                        "L/min",
                    )
                    meanQpv = time_average(tpt, Qpv)
                    print(
                        "Mean Qpv = ",
                        meanQpv * 60,
                        "L/min; Peak Qpv = ",
                        max(Qpv) * 60,
                        "L/min",
                    )

    hemodynamics_outdirectory = os.path.join(
        IODet["outputfolder"], IODet["caseID"], "hemodynamics"
    )
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    # plt.plot(homo_LVV, homo_LVP)
    plt.legend()
    plt.ylabel("Flow rate (L/min)")
    plt.xlabel("Time (s)")
    plt.savefig(os.path.join(hemodynamics_outdirectory, "Flow.png"))
    plt.clf()


def time_average(time_array, data_array):
    """Calculates the time-weighted average of a data array.

    Args:
        time_array (list or numpy array): Array of timestamps.
        data_array (list or numpy array): Array of data values.

    Returns:
        float: Time-weighted average.
    """

    if len(time_array) != len(data_array):
        raise ValueError("Time and data arrays must have the same length.")

    time_array = np.array(time_array)
    data_array = np.array(data_array)

    # Calculate time differences
    time_diffs = np.diff(time_array)

    # Calculate weighted average
    weighted_sum = np.sum(data_array[:-1] * time_diffs)
    total_time = time_array[-1] - time_array[0]

    return weighted_sum / total_time


def plot_PV_comparison(IODet, SimDet, cycle=None):
    """Compare pressure-time curves from two different BiV_PV.txt files."""
    BCL = SimDet["HeartBeatLength"]
    block = IODet["block"]
    
    if cycle is None:
        cycle = SimDet["closedloopparam"]["stop_iter"] +1

    plt.figure()
    # Plot first dataset
    for ncycle in range(cycle):
        filename1 = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/BiV_PV.txt")
        time1, LVP1, _, _, _, _ = extract_PV(filename1, BCL, ncycle, SimDet)
        plt.plot(time1, LVP1)

    for ncycle in range(cycle):
        filename1 = os.path.join(IODet["outputfolder"]+IODet["caseID"]+"/BiV_PV.txt")
        time1, LVP1, _, _, _, _ = extract_PV(filename1, BCL, ncycle, SimDet)
        plt.plot(time1, LVP1,linestyle="dashed")


    # Create output directory if not exists
    hemodynamics_outdirectory = os.path.join(IODet["outputfolder"]+block)
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    # plt.legend()
        # plt.legend()
    plt.axvline(x=50, linestyle='--', color='gray', linewidth=1.2)
    plt.axvline(x=149, linestyle='--', color='gray', linewidth=1.2)
    plt.axvline(x=479, linestyle='--', color='gray', linewidth=1.2)
    plt.axvline(x=610, linestyle='--', color='gray', linewidth=1.2)

    plt.ylabel("Pressure (mmHg)")
    plt.xlabel("Time [ms]")
    # plt.title("Comparison of Baseline and LBBB Pressure")"_cycle_"+str(cycle)+".png"
    plt.savefig(os.path.join(hemodynamics_outdirectory, "Pressure_Comparison_"+block+"_simulation.png"))
    plt.show()


def plot_PV_comparison_ind(IODet, SimDet, cycle=None):
    """Compare pressure-time curves from two different BiV_PV.txt files."""
    BCL = SimDet["HeartBeatLength"]
    ncycle = cycle
    block = IODet["block"]
    crt = IODet["caseID"]

    plt.figure()

    filename1 = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/BiV_PV.txt")
    time1, LVP1, _, _, _, _ = extract_PV(filename1, BCL, ncycle, SimDet)   
    plt.plot(time1, LVP1, label=f"Baseline")
    pdb.set_trace()
    filename2 = os.path.join(IODet["outputfolder"]+IODet["caseID"]+"/BiV_PV.txt")
    time2, LVP2, _, _, _, _ = extract_PV(filename2, BCL, ncycle, SimDet)
    plt.plot(time2, LVP2, linestyle="dashed", label=f"{crt}")    

    filename3 = os.path.join(IODet["outputfolder"]+IODet["block"]+"/BiV_PV.txt")
    time3, LVP3, _, _, _, _ = extract_PV(filename3, BCL, ncycle, SimDet)
    plt.plot(time3, LVP3, linestyle="dashed", label=f"{block}")        

    # Create output directory if not exists
    hemodynamics_outdirectory = os.path.join(IODet["outputfolder"]+block)
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    plt.legend()
    plt.ylabel("Pressure (mmHg)")
    plt.xlabel("Time (s)")
    plt.savefig(os.path.join(hemodynamics_outdirectory, "Pressure_Comparison_"+block+"_cycle_"+str(cycle)+".png"))
    plt.show()    


def plothemodynamics_comparison(IODet, SimDet, cycle=None):
    """Compare PV loops from two different BiV_PV.txt files."""
    BCL = SimDet["HeartBeatLength"]
    block = IODet["block"]    
    
    if cycle is None:
        cycle = SimDet["closedloopparam"]["stop_iter"]+1

    plt.figure()
    # Plot first dataset (solid lines)
    for ncycle in range(cycle):
        filename1 = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/BiV_PV.txt")
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(filename1, BCL, ncycle, SimDet)
        plt.plot(homo_LVV1, homo_LVP1)

    for ncycle in range(cycle):
        filename1 = os.path.join(IODet["outputfolder"]+IODet["caseID"]+"/BiV_PV.txt")
        time1, LVP1, _, _, _, _ = extract_PV(filename1, BCL, ncycle, SimDet)
        plt.plot(time1, LVP1,linestyle="dashed")

    # Create output directory if not exists
    hemodynamics_outdirectory = os.path.join(IODet["outputfolder"]+block)
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    # plt.legend()
    # plt.axvline(x=50, linestyle='--', color='gray', linewidth=1.2)
    # plt.axvline(x=149, linestyle='--', color='gray', linewidth=1.2)
    # plt.axvline(x=479, linestyle='--', color='gray', linewidth=1.2)
    # plt.axvline(x=810, linestyle='--', color='gray', linewidth=1.2)

    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    # plt.title("Comparison of Baseline and LBBB PV Loops") , 
    plt.savefig(os.path.join(hemodynamics_outdirectory, "PV_Comparison_"+block+"_simulation.png"))
    plt.show()


def plothemodynamics_comparison_ind(IODet, SimDet, cycle=None):
    """Compare PV loops from two different BiV_PV.txt files."""
    BCL = SimDet["HeartBeatLength"]
    ncycle = cycle  
    block = IODet["block"]
    crt = IODet["caseID"]

    plt.figure()
    filename1 = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/BiV_PV.txt")
    homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(filename1, BCL, ncycle, SimDet)
    plt.plot(homo_LVV1, homo_LVP1, label=f"Baseline")

    filename2 = os.path.join(IODet["outputfolder"]+IODet["caseID"]+"/BiV_PV.txt")
    homo_tptt2, homo_LVP2, homo_LVV2, homo_RVP2, homo_RVV2, homo_Qmv2 = extract_PV(filename2, BCL, ncycle, SimDet)
    plt.plot(homo_LVV2, homo_LVP2, label=f"{crt}")

    filename3 = os.path.join(IODet["outputfolder"]+IODet["block"]+"/BiV_PV.txt")
    homo_tptt3, homo_LVP3, homo_LVV3, homo_RVP3, homo_RVV3, homo_Qmv3 = extract_PV(filename3, BCL, ncycle, SimDet)
    plt.plot(homo_LVV3, homo_LVP3, label=f"{block}")

    # Create output directory if not exists
    hemodynamics_outdirectory = os.path.join(IODet["outputfolder"]+block)
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    plt.legend()
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    
    # plt.title(f"Comparison of Baseline and LBBB PV Loops") , 
    plt.savefig(os.path.join(hemodynamics_outdirectory, "PV_Comparison_"+block+"_cycle_"+str(cycle)+".png"))
    plt.show()    

# def rotation_matrix_z(theta):
#     R = np.eye(3)
#     R[0, 0] = np.cos(theta)
#     R[0, 1] = -np.sin(theta)
#     R[1, 0] = np.sin(theta)
#     R[1, 1] = np.cos(theta)
#     return R

# def rotate_vector_function(evec, R):
#     V = evec.function_space()
#     dofmap = V.dofmap()
#     mesh = V.mesh()
#     vec = evec.vector().get_local()
#     new_vec = np.zeros_like(vec)

#     for cell in df.cells(mesh):
#         dofs = dofmap.cell_dofs(cell.index())
#         for i in range(0, len(dofs), 3):
#             v = vec[dofs[i:i+3]]
#             v_rot = R @ v
#             new_vec[dofs[i:i+3]] = v_rot
#     evec.vector().set_local(new_vec)
#     evec.vector().apply("insert")

def compute_strain_AHA_profile(IODet, SimDet, LVid=1, cycle=None, R_apex=1.0):
    isLV = False
    isBiV = False
    BCL = SimDet["HeartBeatLength"]
    block = IODet["block"]  
    t_vol_min = np.zeros(cycle+1)   

    for ncycle in range(cycle+1):
        filename1 = os.path.join(IODet["outputfolder"]+IODet["caseID"]+"/BiV_PV.txt")
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(filename1, BCL, ncycle, SimDet)
        t_vol_min[ncycle] = homo_tptt1[np.argmin(homo_LVV1)]

    mesh = df.Mesh()
    hdf = df.HDF5File(mesh.mpi_comm(),IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5","r",)
    u_arr = extractdisplacement(IODet, SimDet, cycle=None)

    if "isLV" in list(SimDet.keys()):
        isLV = SimDet["isLV"]

        if(isLV):
            mesh_params = {
                "directory": IODet["directory_me"],
                "casename": IODet["casename_me"],
                "outputfolder": IODet["outputfolder"],
                "foldername": IODet["folderName"],
                "isLV": isLV,
            }

            Mesh_obj = lv_mechanics_mesh(mesh_params, SimDet)

            try:
                eC0 = Mesh_obj.eC0
                eL0 = Mesh_obj.eL0
                eR0 = Mesh_obj.eR0

            except AttributeError:
                fiber_angle_param = {   "mesh": Mesh_obj.mesh,
                                        "facetboundaries": Mesh_obj.facetboundaries,
                                        "LV_fiber_angle": [0.01, -0.01],
                                        "LV_sheet_angle": [0.1, -0.1],
                                        "minztol": Mesh_obj.mesh.hmax()/2.0, # Coarse mesh
                                        "isrotatept": False,
                                        "isreturn": True,
                                        "outfilename": IODet["casename_me"],
                                        "outdirectory": IODet["outputfolder"] + IODet["caseID"] + IODet["folderName"],
                                        "baseid": SimDet["topid"],
                                        "epiid": SimDet["epiid"],
                                        "lvid": SimDet["LVendoid"],
                                        "degree": SimDet["GiccioneParams"]["deg"]}

                eC0, eL0, eR0  = vtk_py.addLVfiber_LDRB(fiber_angle_param)

    deg = SimDet["GiccioneParams"]["deg"]

    # # Optional: rotate fiber vectors using HIS bundle
    # if "HIS" in SimDet and SimDet["HIS"] is not None:
    #     his = SimDet["HIS"]
    #     his = np.array(his).flatten()
    #     # pdb.set_trace()
    #     theta = np.pi - np.arctan2(his[1], his[0])
    #     R = rotation_matrix_z(theta)
    #     rotate_vector_function(eC0, R)
    #     rotate_vector_function(eL0, R)
    #     rotate_vector_function(eR0, R)

    # Normalize directions
    eC0_normalized = eC0 / df.sqrt(df.inner(eC0, eC0))
    eL0_normalized = eL0 / df.sqrt(df.inner(eL0, eL0))
    eR0_normalized = eR0 / df.sqrt(df.inner(eR0, eR0))

    aha_segments = tag_AHA_segments(Mesh_obj.mesh, apex_axis="z", R_apex=1.0, reference_point=SimDet["HIS"], rotate=True, plot_debug=False) 
    dx_aha = df.Measure("dx", domain=Mesh_obj.mesh, subdomain_data=aha_segments)

    if SimDet["Mechanics Discretization"] is "P1P1":
        var_deg = 1
    else:
        var_deg = 2

    # === Setup spaces and strain computation ===
    udisp = df.Function(df.VectorFunctionSpace(Mesh_obj.mesh, "CG", var_deg))
    udisp.vector()[:] = u_arr[0].vector().get_local()[:]

    GuccioneParams = SimDet["GiccioneParams"]
    params = {  "mesh": Mesh_obj.mesh,
                "displacement_variable": udisp,
                "material model": SimDet["GiccioneParams"]["Passive model"],
                "material params": SimDet["GiccioneParams"]["Passive params"],
                "incompressible": SimDet["GiccioneParams"]["incompressible"],
                "growth_tensor": None,}

    uflforms = Forms(params)
    Fref = df.project(uflforms.Fmat(), df.TensorFunctionSpace(Mesh_obj.mesh, "DG", 0))

    # === Storage ===
    n_seg = 17
    Ecc_arr = [[] for _ in range(n_seg)]
    Ell_arr = [[] for _ in range(n_seg)]
    # Err_arr = [[] for _ in range(n_seg)]

    # === Loop over time ===
    for u_t in u_arr:
        udisp.vector()[:] = u_t.vector().get_local()[:]
        Fmat = uflforms.Fmat()
        F = Fmat * df.inv(Fref)
        Cmat = F.T * F

        Ccc = df.inner(eC0_normalized, Cmat * eC0_normalized)
        Cll = df.inner(eL0_normalized, Cmat * eL0_normalized)  
        # Crr = df.inner(eR0_normalized, Cmat * eR0_normalized)    

        Ecc = 0.5 * (1 - 1 / Ccc)      
        Ell = 0.5 * (1 - 1 / Cll)        
        # Err = 0.5 * (1 - 1 / Crr)

        volumes = [df.assemble(df.Constant(1.0) * dx_aha(seg)) for seg in range(1, 18)]

        for seg in range(1, n_seg + 1):
            vol = volumes[seg - 1]

            if vol > 0:
                Ecc_arr[seg - 1].append(df.assemble(Ecc * dx_aha(seg)) / vol)
                Ell_arr[seg - 1].append(df.assemble(Ell * dx_aha(seg)) / vol)
                # Err_arr[seg - 1].append(df.assemble(Err * dx_aha(seg)) / vol)
            else:
                Ecc_arr[seg - 1].append(np.nan)
                Ell_arr[seg - 1].append(np.nan)
                # Err_arr[seg - 1].append(np.nan)

    # === Save output ===
    outdir = os.path.join(IODet["outputfolder"], IODet["caseID"])
    np.savez(os.path.join(outdir, "Ecc_AHA.npz"), *Ecc_arr)
    np.savez(os.path.join(outdir, "Ell_AHA.npz"), *Ell_arr)
    # np.savez(os.path.join(outdir, "Err_AHA.npz"), *Err_arr)


# def tag_AHA_segments(mesh, apex_axis="z", R_apex=1.0, reference_point=None, rotate=True, plot_debug=False):
#     """
#     Tag AHA segments on the mesh. Optionally rotates based on HIS bundle direction.
#     If plot_debug is True, generates a plot showing segment angle assignments and reference vector.
#     """
#     mesh.init()
#     cell_tags = df.MeshFunction("size_t", mesh, mesh.topology().dim())
#     cell_tags.set_all(0)

#     coords = mesh.coordinates()
#     z_vals = coords[:, 2]
#     z_min, z_max = z_vals.min(), z_vals.max()
#     h = z_max - z_min

#     basal_th = z_max - h / 3
#     mid_th = z_max - 2 * h / 3

#     theta_rot = 0.0
#     his_length = 1.0
#     if reference_point and rotate:
#         ref_vec = np.array(reference_point).flatten()
#         if ref_vec.ndim == 1 and len(ref_vec) >= 2:
#             his_length = np.linalg.norm(ref_vec[:2])
#             theta_rot = np.pi - np.arctan2(ref_vec[1], ref_vec[0])  # Rotate HIS to align with -x axis
#             for i in range(len(coords)):
#                 x, y = coords[i][0], coords[i][1]
#                 x_new = np.cos(theta_rot) * x - np.sin(theta_rot) * y
#                 y_new = np.sin(theta_rot) * x + np.cos(theta_rot) * y
#                 coords[i][0] = x_new
#                 coords[i][1] = y_new

#     debug_coords = []
#     debug_angles = []
#     debug_levels = []

#     for cell in df.cells(mesh):
#         mp = cell.midpoint()
#         x, y, z = mp.x(), mp.y(), mp.z()

#         theta = np.arctan2(y, x)
#         theta_deg = np.degrees(theta) % 360

#         if z >= basal_th:
#             level = "basal"
#         elif z >= mid_th:
#             level = "mid"
#         else:
#             level = "apical"

#         # Uniformly rotate all levels by 150° so that segment 6 aligns with -x
#         theta_shifted = (theta_deg - 60) % 360
#         theta_shifted_ap = (theta_deg - 45) % 360

#         if level == "basal":
#             tag = int(theta_shifted // 60) + 1
#         elif level == "mid":
#             tag = int(theta_shifted // 60) + 7
#         elif level == "apical":
#             tag = int(theta_shifted_ap // 90) + 13

#         if np.linalg.norm([x, y, z - z_min]) < R_apex:
#             tag = 17

#         cell_tags[cell.index()] = tag

#         if plot_debug:
#             debug_coords.append([x, y])
#             debug_angles.append(theta_deg)
#             debug_levels.append(level)

#     if plot_debug:
#         debug_coords = np.array(debug_coords)
#         fig, ax = plt.subplots(figsize=(6, 6))
#         ax.scatter(debug_coords[:, 0], debug_coords[:, 1], c=debug_angles, cmap='hsv', s=10)
#         ax.set_title("AHA Tagging Polar Angles")
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.axis('equal')

#         if reference_point:
#             ax.quiver(0, 0, ref_vec[0], ref_vec[1], color='red', scale=1, scale_units='xy', 
#                       angles='xy', width=0.01, label="HIS vector (before rotation)")
#             ax.quiver(0, 0, -his_length, 0, color='blue', scale=1, scale_units='xy', angles='xy', 
#                       width=0.01, label="Target -x direction (segment 1 dir)")
#             ax.legend()

#         plt.grid(True)
#         plt.show()

#     return cell_tags

def tag_AHA_segments(mesh, apex_axis="z", R_apex=1.0, reference_point=None, rotate=True, plot_debug=False):
    mesh.init()
    cell_tags = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    cell_tags.set_all(0)

    coords = mesh.coordinates()
    z_vals = coords[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    h = z_max - z_min
    basal_th = z_max - h / 3
    mid_th = z_max - 2 * h / 3

    # Compute theta_shift to align segment 2 with HIS
    theta_shift = 0.0
    if reference_point and rotate:
        his = np.array(reference_point).flatten()
        theta_his = np.degrees(np.arctan2(his[1], his[0])) % 360
        theta_shift = (theta_his - 330) % 360  # Align end of segment 2 with HIS

    for cell in df.cells(mesh):
        mp = cell.midpoint()
        x, y, z = mp.x(), mp.y(), mp.z()

        theta = np.degrees(np.arctan2(y, x)) % 360
        theta_rotated = (theta + theta_shift) % 360

        if z >= basal_th:
            level = "basal"
            tag = int((theta_rotated - 60) % 360 // 60) + 1
        elif z >= mid_th:
            level = "mid"
            tag = int((theta_rotated - 60) % 360 // 60) + 7
        else:
            level = "apical"
            tag = int((theta_rotated - 45) % 360 // 90) + 13

        if np.linalg.norm([x, y, z - z_min]) < R_apex:
            tag = 17

        cell_tags[cell.index()] = tag

    return cell_tags


def bullseye_plot(ax, data, seg_bold=None, cmap="Greys", norm=None):
    data = np.ravel(data)
    data = np.ma.masked_array(data, mask=[False]*16 + [True])

    cmap.set_bad(color='black')

    if seg_bold is None:
        seg_bold = []
    if norm is None:
        norm = Normalize(vmin=data.min(), vmax=data.max())

    r = np.linspace(0.2, 1, 4)
    ax.set(ylim=[0, 1], xticklabels=[], yticklabels=[])
    ax.grid(False)

    for start, stop, r_in, r_out in [
        (0, 6, r[2], r[3]),
        (6, 12, r[1], r[2]),
        (12, 16, r[0], r[1]),
        (16, 17, 0, r[0]),
    ]:
        n = stop - start
        dtheta = 2*np.pi / n
        ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
               color=cmap(norm(data[start:stop])))

    for start, stop, r_in, r_out in [
        (0, 6, r[2], r[3]),
        (6, 12, r[1], r[2]),
        (12, 16, r[0], r[1]),
    ]:
        n = stop - start
        dtheta = 2*np.pi / n
        ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
               clip_on=False, color="none", edgecolor="black", linewidth=[
                   4 if i + 1 in seg_bold else 1 for i in range(start, stop)])
    ax.plot(np.linspace(0, 2*np.pi), np.linspace(r[0], r[0]), "black",
            linewidth=(4 if 17 in seg_bold else 1))


def check_AHA_plot(IODet, SimDet, target_segment=1, R_apex=3.0):
    """
    Highlight segments 1 and 2 in bullseye plot and in a VTK file.
    """
    # === Load mesh ===
    mesh_params = {
        "directory": IODet["directory_me"],
        "casename": IODet["casename_me"],
        "outputfolder": IODet["outputfolder"],
        "foldername": IODet["folderName"],
        "isLV": SimDet.get("isLV", True),
    }
    Mesh_obj = lv_mechanics_mesh(mesh_params, SimDet)
    mesh = Mesh_obj.mesh
    # === Save bullseye plot with segments 1 and 2 highlighted ===
    outdir = os.path.join(IODet["outputfolder"], IODet["caseID"])
    os.makedirs(outdir, exist_ok=True)

    # === Tag AHA segments ===
    # aha_segments = tag_AHA_segments(mesh, R_apex=R_apex, reference_point=SimDet["HIS"])
    aha_segments = tag_AHA_segments(mesh, R_apex=R_apex, reference_point=SimDet["HIS"], rotate=True, plot_debug=True)

    # === Segment counts ===
    num_cells = mesh.num_cells()
    segment_counts = np.zeros(17, dtype=int)
    for cell in df.cells(mesh):
        tag = aha_segments[cell.index()]
        if 1 <= tag <= 17:
            segment_counts[tag - 1] += 1


    # binary_data = np.zeros(17)
    # binary_data[0] = 1.0  # segment 1
    # binary_data[1] = 1.0  # segment 2
    # fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    # bullseye_plot(ax, data=binary_data, seg_bold=[1, 2],
    #               cmap=cm.get_cmap("Greys"), norm=mpl.colors.Normalize(vmin=0, vmax=1))
    # plt.savefig(os.path.join(outdir, "AHA_bullseye_seg1and2.png"))
    # plt.close()

    # === Create VTK field for segments 1 and 2 ===
    V0 = df.FunctionSpace(mesh, "DG", 0)
    mask = df.Function(V0)
    # mask_values = np.zeros(V0.dim())
    # for cell in df.cells(mesh):
    #     tag = aha_segments[cell.index()]
    #     if tag == 1:
    #         mask_values[cell.index()] = 1.0
    #     elif tag == 2:
    #         mask_values[cell.index()] = 2.0
    # mask.vector().set_local(mask_values)
    # mask.vector().apply("insert")

    # vtkpath = os.path.join(outdir, "AHA_segments_1_and_2_mask.pvd")
    # df.File(vtkpath) << mask
    mask_values = mask.vector().get_local()

    for cell in df.cells(mesh):
        tag = aha_segments[cell.index()]
        mask_values[cell.index()] = tag  # Integer tag directly

    mask.vector().set_local(mask_values)
    mask.vector().apply("insert")

    # Save all segment tags
    vtkpath = os.path.join(outdir, "AHA_segments_all_tags.pvd")
    df.File(vtkpath) << mask

    print(f"✅ All AHA segment tags written to: {vtkpath}")    

    # print(f"✅ Bullseye plot saved: {os.path.join(outdir, 'AHA_bullseye_seg1and2.png')}")
    # print(f"✅ VTK mask saved: {vtkpath}")


def plot_min_bullseye_per_cycle(IODet, SimDet):
    """
    Plots the minimum strain per AHA segment for each cycle using bullseye plots.

    Parameters
    ----------
    IODet : dict
        Dictionary containing I/O details like 'caseID', 'outputfolder'.
    SimDet : dict
        Dictionary containing simulation details.
    """
    outdir = os.path.join(IODet["outputfolder"], IODet["caseID"])
    ecc_path = os.path.join(outdir, "Ecc_AHA.npz")
    ell_path = os.path.join(outdir, "Ell_AHA.npz")    
    ncycle = SimDet["closedloopparam"]["stop_iter"] + 1
    BCL = SimDet["HeartBeatLength"]    
    pv_file = os.path.join(IODet["outputfolder"]+IODet["caseID"]+"/BiV_PV.txt")     
    t_systole= np.zeros(ncycle)   
   
    for cycle in range(ncycle):
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(pv_file, BCL, cycle, SimDet)
        t_systole[cycle] = homo_tptt1[np.argmin(homo_LVV1)]

    if not os.path.exists(ecc_path):
        raise FileNotFoundError(f"Ecc file not found: {ecc_path}")

    if not os.path.exists(ell_path):
        raise FileNotFoundError(f"Ell file not found: {ell_path}")    

    # Load data
    data_ecc = np.load(ecc_path)
    n_segments_ecc = len(data_ecc.files)
    segment_strains_ecc = [data_ecc[f'arr_{i}'] for i in range(n_segments_ecc)]
    strain_matrix_ecc = np.vstack(segment_strains_ecc)  # shape: (17, N)
    n_timesteps_ecc = strain_matrix_ecc.shape[1]

    data_ell = np.load(ell_path)
    n_segments_ell = len(data_ell.files)
    segment_strains_ell = [data_ell[f'arr_{i}'] for i in range(n_segments_ell)]
    strain_matrix_ell = np.vstack(segment_strains_ell)  # shape: (17, N)
    n_timesteps_ell = strain_matrix_ell.shape[1]

    if n_timesteps_ecc % ncycle != 0:
        raise ValueError("Strain data does not contain correct number of time steps.")

    frames_per_cycle = n_timesteps_ecc // ncycle

    sdi_ecc_all = []
    sdi_ell_all = []   

    for cycle in range(ncycle):
        start = cycle * frames_per_cycle
        end = (cycle + 1) * frames_per_cycle

        min_vals_ecc = np.min(strain_matrix_ecc[:, start:end], axis=1)
        min_vals_ell = np.min(strain_matrix_ell[:, start:end], axis=1)

         # --- Save strain minima and time indices ---
        min_ecc_times = np.argmin(strain_matrix_ecc[:, start:end], axis=1) + start
        min_ell_times = np.argmin(strain_matrix_ell[:, start:end], axis=1) + start

        # convert_frame_index_to_time(min_ecc_times)
        min_ecc_time_ms = convert_frame_index_to_time(min_ecc_times, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])
        min_ell_time_ms = convert_frame_index_to_time(min_ell_times, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])

        valid_min_ecc_times_ms = min_ecc_time_ms[:16]  # segments 1–16 only
        valid_min_ell_times_ms = min_ell_time_ms[:16]  # segments 1–16 only

        print(min_ecc_time_ms)
        print(min_ell_time_ms)
        print(t_systole[cycle])
        print(valid_min_ecc_times_ms - (t_systole[cycle]))
        print(valid_min_ell_times_ms - (t_systole[cycle]))        

        # if cycle == 0:
        SDI_Ecc = np.std(valid_min_ecc_times_ms - (t_systole[cycle])) / BCL * 100
        SDI_Ell = np.std(valid_min_ell_times_ms - (t_systole[cycle])) / BCL * 100
        # else:
        #     pdb.set_trace()
        #     SDI_Ecc = np.std(valid_min_ecc_times_ms - (t_systole[cycle]-(cycle-1)*BCL)) / BCL * 100
        #     SDI_Ell = np.std(valid_min_ell_times_ms - (t_systole[cycle]-(cycle-1)*BCL)) / BCL * 100    

        sdi_ecc_all.append(SDI_Ecc)
        sdi_ell_all.append(SDI_Ell)

        print(f"Ecc SDI: {SDI_Ecc:.4f}")
        print(f"Ell SDI: {SDI_Ell:.4f}")

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ecc = np.full(17, np.nan)
        delta_time_ecc[:16] = valid_min_ecc_times_ms - (t_systole[cycle] if cycle == 0 else t_systole[cycle] - (cycle - 1) * BCL)
        # delta_time_ecc[16] = np.nan  # Segment 17 (center) white

        # Normalize color scale to this delta
        vmin_dt, vmax_dt = np.nanmin(delta_time_ecc[:16]), np.nanmax(delta_time_ecc[:16])
        norm_dt = mpl.colors.Normalize(vmin=vmin_dt, vmax=vmax_dt)
        cmap_dt = cm.coolwarm

        fig_dt, ax_dt = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        bullseye_plot(ax_dt, data=delta_time_ecc, seg_bold=[], cmap=cmap_dt, norm=norm_dt)

        sm_dt = mpl.cm.ScalarMappable(cmap=cmap_dt, norm=norm_dt)
        sm_dt.set_array([])
        cbar_dt = plt.colorbar(sm_dt, ax=ax_dt, orientation='horizontal', pad=0.05, shrink=0.6)
        cbar_dt.set_label("ΔT: Min Ecc vs Systole [ms]", fontsize=12)
        ax_dt.set_title(f"ΔT (Ecc min - systole) - Cycle {cycle + 1}", fontsize=14)

        delta_fig_path = os.path.join(outdir, f"delta_minEcc_from_systole_cycle{cycle+1}.png")
        plt.savefig(delta_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig_dt)
        print(f"Saved: {delta_fig_path}")

        # --- Bullseye of time delta between min Ell and t_systole ---
        delta_time_ell = np.full(17, np.nan)
        delta_time_ell[:16] = valid_min_ell_times_ms - (t_systole[cycle] if cycle == 0 else t_systole[cycle] - (cycle - 1) * BCL)
        # delta_time_ell[16] = np.nan  # Segment 17 (center) white
        
        # Normalize color scale to this delta
        vmin_dt, vmax_dt = np.nanmin(delta_time_ell[:16]), np.nanmax(delta_time_ell[:16])
        norm_dt = mpl.colors.Normalize(vmin=vmin_dt, vmax=vmax_dt)
        cmap_dt = cm.coolwarm

        fig_dt, ax_dt = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        bullseye_plot(ax_dt, data=delta_time_ell, seg_bold=[], cmap=cmap_dt, norm=norm_dt)

        sm_dt = mpl.cm.ScalarMappable(cmap=cmap_dt, norm=norm_dt)
        sm_dt.set_array([])
        cbar_dt = plt.colorbar(sm_dt, ax=ax_dt, orientation='horizontal', pad=0.05, shrink=0.6)
        cbar_dt.set_label("ΔT: Min Ell vs Systole [ms]", fontsize=12)
        ax_dt.set_title(f"ΔT (Ell min - systole) - Cycle {cycle + 1}", fontsize=14)

        delta_fig_path = os.path.join(outdir, f"delta_minEll_from_systole_cycle{cycle+1}.png")
        plt.savefig(delta_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig_dt)
        print(f"Saved: {delta_fig_path}")

        # pdb.set_trace()
        # --- Full bullseye with strain values ---
        vmin_ecc, vmax_ecc = np.nanmin(min_vals_ecc), np.nanmax(min_vals_ecc)
        norm = mpl.colors.Normalize(vmin=vmin_ecc, vmax=vmax_ecc)
        cmap = cm.coolwarm

        fig1, ax1 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        bullseye_plot(ax1, data=min_vals_ecc, seg_bold=[], cmap=cmap, norm=norm)

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.6)
        cbar.set_label("Minimum circumferential Strain", fontsize=12)
        ax1.set_title(f"Min Ecc Strain - Cycle {cycle + 1}", fontsize=14)

        fullpath = os.path.join(outdir, f"min_Ecc_bullseye_cycle{cycle+1}.png")
        plt.savefig(fullpath, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"Saved: {fullpath}")

        # --- Full bullseye with strain values ---
        vmin_ell, vmax_ell = np.nanmin(min_vals_ell), np.nanmax(min_vals_ell)
        norm = mpl.colors.Normalize(vmin=vmin_ell, vmax=vmax_ell)
        cmap = cm.coolwarm

        fig2, ax2 = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        bullseye_plot(ax2, data=min_vals_ell, seg_bold=[], cmap=cmap, norm=norm)

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.4)
        cbar.set_label("Minimum longitudinal Strain", fontsize=12)
        ax2.set_title(f"Min Ell Strain - Cycle {cycle + 1}", fontsize=14)

        fullpath = os.path.join(outdir, f"min_Ell_bullseye_cycle{cycle+1}.png")
        plt.savefig(fullpath, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved: {fullpath}")

    # After the loop
    with open(os.path.join(outdir, "SDI_summary.txt"), "w") as f:
        f.write("Cycle\tSDI_Ecc (%)\tSDI_Ell (%)\n")
        for i, (ecc_val, ell_val) in enumerate(zip(sdi_ecc_all, sdi_ell_all), 1):
            f.write(f"{i}\t{ecc_val:.4f}\t{ell_val:.4f}\n")


def convert_frame_index_to_time(min_Ecc_time_array, dt, step_interval, t0):
    """
    Converts an array of frame indices to time in ms.
    """
    return np.array([
         (t0 + dt *step_interval * idx) if idx >= 0 else -1
        for idx in min_Ecc_time_array
    ])     

def strain_plot_AHA(IODet,SimDet,mode='all'):
    #directory
    outdir = os.path.join(IODet["outputfolder"], IODet["caseID"])

    data_ecc = np.load(outdir+"/Ecc_AHA.npz")  
    data_ell = np.load(outdir+"/Ell_AHA.npz")     

    ###### data
    ecc_array1 = data_ecc['arr_0'].flatten()
    ecc_array2 = data_ecc['arr_1'].flatten()
    ecc_array3 = data_ecc['arr_2'].flatten()
    ecc_array4 = data_ecc['arr_3'].flatten()
    ecc_array5 = data_ecc['arr_4'].flatten()
    ecc_array6 = data_ecc['arr_5'].flatten()
    ecc_array7 = data_ecc['arr_6'].flatten()
    ecc_array8 = data_ecc['arr_7'].flatten()
    ecc_array9 = data_ecc['arr_8'].flatten()
    ecc_array10 = data_ecc['arr_9'].flatten()
    ecc_array11 = data_ecc['arr_10'].flatten()
    ecc_array12 = data_ecc['arr_11'].flatten()
    ecc_array13 = data_ecc['arr_12'].flatten()
    ecc_array14 = data_ecc['arr_13'].flatten()
    ecc_array15 = data_ecc['arr_14'].flatten()
    ecc_array16 = data_ecc['arr_15'].flatten()
    ecc_array17 = data_ecc['arr_16'].flatten()

    ell_array1 = data_ell['arr_0'].flatten()
    ell_array2 = data_ell['arr_1'].flatten()
    ell_array3 = data_ell['arr_2'].flatten()
    ell_array4 = data_ell['arr_3'].flatten()
    ell_array5 = data_ell['arr_4'].flatten()
    ell_array6 = data_ell['arr_5'].flatten()
    ell_array7 = data_ell['arr_6'].flatten()
    ell_array8 = data_ell['arr_7'].flatten()
    ell_array9 = data_ell['arr_8'].flatten()
    ell_array10 = data_ell['arr_9'].flatten()
    ell_array11 = data_ell['arr_10'].flatten()
    ell_array12 = data_ell['arr_11'].flatten()
    ell_array13 = data_ell['arr_12'].flatten()
    ell_array14 = data_ell['arr_13'].flatten()
    ell_array15 = data_ell['arr_14'].flatten()
    ell_array16 = data_ell['arr_15'].flatten()
    ell_array17 = data_ell['arr_16'].flatten()


    if mode == 'all':
        plt.figure()
        plt.plot(np.arange(0, len(ecc_array1)), ecc_array1)
        plt.plot(np.arange(0, len(ecc_array2)), ecc_array2)
        plt.plot(np.arange(0, len(ecc_array3)), ecc_array3, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array4)), ecc_array4, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array5)), ecc_array5, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array6)), ecc_array6)
        plt.plot(np.arange(0, len(ecc_array7)), ecc_array7)
        plt.plot(np.arange(0, len(ecc_array8)), ecc_array8)
        plt.plot(np.arange(0, len(ecc_array9)), ecc_array9, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array10)), ecc_array10, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array11)), ecc_array11, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array12)), ecc_array12)
        plt.plot(np.arange(0, len(ecc_array13)), ecc_array13)
        plt.plot(np.arange(0, len(ecc_array14)), ecc_array14)
        plt.plot(np.arange(0, len(ecc_array15)), ecc_array15)
        plt.plot(np.arange(0, len(ecc_array16)), ecc_array16)
        plt.plot(np.arange(0, len(ecc_array17)), ecc_array17, linestyle="dotted")
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(outdir,"Ecc_aha.png"))
        plt.clf() 

        plt.figure()
        plt.plot(np.arange(0, len(ell_array1)), ell_array1)
        plt.plot(np.arange(0, len(ell_array2)), ell_array2)
        plt.plot(np.arange(0, len(ell_array3)), ell_array3, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array4)), ell_array4, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array5)), ell_array5, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array6)), ell_array6)
        plt.plot(np.arange(0, len(ell_array7)), ell_array7)
        plt.plot(np.arange(0, len(ell_array8)), ell_array8)
        plt.plot(np.arange(0, len(ell_array9)), ell_array9, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array10)), ell_array10, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array11)), ell_array11, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array12)), ell_array12)
        plt.plot(np.arange(0, len(ell_array13)), ell_array13)
        plt.plot(np.arange(0, len(ell_array14)), ell_array14)
        plt.plot(np.arange(0, len(ell_array15)), ell_array15)
        plt.plot(np.arange(0, len(ell_array16)), ell_array16)
        plt.plot(np.arange(0, len(ell_array17)), ell_array17, linestyle="dotted")
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(outdir,"Ell_aha.png"))
        plt.clf()         

    elif mode == 'split':  
        plt.figure()
        plt.plot(np.arange(0, len(ecc_array1)), ecc_array1)
        plt.plot(np.arange(0, len(ecc_array2)), ecc_array2)
        plt.plot(np.arange(0, len(ecc_array3)), ecc_array3, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array4)), ecc_array4, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array5)), ecc_array5, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array6)), ecc_array6)
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(outdir,"Ecc_aha_base.png"))
        plt.clf() 

        plt.figure()
        plt.plot(np.arange(0, len(ell_array1)), ell_array1)
        plt.plot(np.arange(0, len(ell_array2)), ell_array2)
        plt.plot(np.arange(0, len(ell_array3)), ell_array3, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array4)), ell_array4, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array5)), ell_array5, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array6)), ell_array6)
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(outdir,"Ell_aha_base.png"))
        plt.clf()  

        plt.figure()
        plt.plot(np.arange(0, len(ecc_array7)), ecc_array7)
        plt.plot(np.arange(0, len(ecc_array8)), ecc_array8)
        plt.plot(np.arange(0, len(ecc_array9)), ecc_array9, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array10)), ecc_array10, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array11)), ecc_array11, linestyle="dotted")
        plt.plot(np.arange(0, len(ecc_array12)), ecc_array12)
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(outdir,"Ecc_aha_mid.png"))
        plt.clf() 

        plt.figure()
        plt.plot(np.arange(0, len(ell_array7)), ell_array7)
        plt.plot(np.arange(0, len(ell_array8)), ell_array8)
        plt.plot(np.arange(0, len(ell_array9)), ell_array9, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array10)), ell_array10, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array11)), ell_array11, linestyle="dotted")
        plt.plot(np.arange(0, len(ell_array12)), ell_array12) 
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(outdir,"Ell_aha_mid.png"))
        plt.clf()  

        plt.figure()
        plt.plot(np.arange(0, len(ecc_array13)), ecc_array13)
        plt.plot(np.arange(0, len(ecc_array14)), ecc_array14)
        plt.plot(np.arange(0, len(ecc_array15)), ecc_array15)
        plt.plot(np.arange(0, len(ecc_array16)), ecc_array16)
        plt.plot(np.arange(0, len(ecc_array17)), ecc_array17, linestyle="dotted")
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(outdir,"Ecc_aha_apex.png"))
        plt.clf() 

        plt.figure()
        plt.plot(np.arange(0, len(ell_array13)), ell_array13)
        plt.plot(np.arange(0, len(ell_array14)), ell_array14)
        plt.plot(np.arange(0, len(ell_array15)), ell_array15)
        plt.plot(np.arange(0, len(ell_array16)), ell_array16)
        plt.plot(np.arange(0, len(ell_array17)), ell_array17, linestyle="dotted")
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(outdir,"Ell_aha_apex.png"))
        plt.clf()               

def plot_min_bullseye_per_cycle_comparison(IODet, SimDet):
    """
    Plot a 3x3 grid of bullseye plots (17 AHA segments each) with a shared colorbar.

    Parameters
    ----------
    data_list : list of ndarray
        List of 9 numpy arrays, each of shape (17,) representing AHA segment data.
    save_path : str or None
        If provided, the figure will be saved to this path.
    vmin, vmax : float or None
        If provided, sets the color scale limits for all subplots.
    cmap : colormap
        Colormap for all plots.
    titles : list of str
        Optional titles for each subplot.
    """
    ncycle = SimDet["closedloopparam"]["stop_iter"] + 1
    BCL = SimDet["HeartBeatLength"]  
    outputdir = IODet["block"] 

    #################################################################
    ########### baseline
    #################################################################      
    outdir_baseline = os.path.join(IODet["outputfolder"], IODet["caseID"])
    ecc_path_baseline = os.path.join(outdir_baseline, "Ecc_AHA.npz")
    ell_path_baseline = os.path.join(outdir_baseline, "Ell_AHA.npz")    

    pv_file_baseline = os.path.join(IODet["outputfolder"]+IODet["caseID"]+"/BiV_PV.txt")     
    t_systole_baseline= np.zeros(ncycle)   
   
    for cycle in range(ncycle):
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(pv_file_baseline, BCL, cycle, SimDet)
        t_systole_baseline[cycle] = homo_tptt1[np.argmin(homo_LVV1)]    

    # Load data baseline
    data_ecc_baseline = np.load(ecc_path_baseline)
    n_segments_ecc_baseline = len(data_ecc_baseline.files)
    segment_strains_ecc_baseline = [data_ecc_baseline[f'arr_{i}'] for i in range(n_segments_ecc_baseline)]
    strain_matrix_ecc_baseline = np.vstack(segment_strains_ecc_baseline)  # shape: (17, N)
    n_timesteps_ecc_baseline = strain_matrix_ecc_baseline.shape[1]

    data_ell_baseline = np.load(ell_path_baseline)
    n_segments_ell_baseline = len(data_ell_baseline.files)
    segment_strains_ell_baseline = [data_ell_baseline[f'arr_{i}'] for i in range(n_segments_ell_baseline)]
    strain_matrix_ell_baseline = np.vstack(segment_strains_ell_baseline)  # shape: (17, N)
    n_timesteps_ell_baseline = strain_matrix_ell_baseline.shape[1]
    frames_per_cycle_baseline = n_timesteps_ecc_baseline // ncycle

    #################################################################
    ########### lafb
    #################################################################    
    outdir_lafb = "/mnt/home/hagersan/GITLAB/heArt/demo/LBBB-fine/LVelectromechanics-PJ-lbbb8-new"
    ecc_path_lafb = os.path.join(outdir_lafb, "Ecc_AHA.npz")
    ell_path_lafb = os.path.join(outdir_lafb, "Ell_AHA.npz") 

    pv_file_lafb = os.path.join(outdir_lafb+"/BiV_PV.txt")     
    t_systole_lafb= np.zeros(ncycle)   
   
    for cycle in range(ncycle):
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(pv_file_lafb, BCL, cycle, SimDet)
        t_systole_lafb[cycle] = homo_tptt1[np.argmin(homo_LVV1)] 

    # Load data lafb
    data_ecc_lafb = np.load(ecc_path_lafb)
    n_segments_ecc_lafb = len(data_ecc_lafb.files)
    segment_strains_ecc_lafb = [data_ecc_lafb[f'arr_{i}'] for i in range(n_segments_ecc_lafb)]
    strain_matrix_ecc_lafb = np.vstack(segment_strains_ecc_lafb)  # shape: (17, N)
    n_timesteps_ecc_lafb = strain_matrix_ecc_lafb.shape[1]

    data_ell_lafb = np.load(ell_path_lafb)
    n_segments_ell_lafb = len(data_ell_lafb.files)
    segment_strains_ell_lafb = [data_ell_lafb[f'arr_{i}'] for i in range(n_segments_ell_lafb)]
    strain_matrix_ell_lafb = np.vstack(segment_strains_ell_lafb)  # shape: (17, N)
    n_timesteps_ell_lafb = strain_matrix_ell_lafb.shape[1]
    frames_per_cycle_lafb = n_timesteps_ecc_lafb // ncycle

    #################################################################
    ########### septal-apical block
    #################################################################    
    outdir_sa = "/mnt/home/hagersan/GITLAB/heArt/demo/LBBB-fine/LVelectromechanics-PJ-lbbb398-new"
    ecc_path_sa = os.path.join(outdir_sa, "Ecc_AHA.npz")
    ell_path_sa = os.path.join(outdir_sa, "Ell_AHA.npz") 

    pv_file_sa = os.path.join(outdir_sa+"/BiV_PV.txt")     
    t_systole_sa= np.zeros(ncycle)   
   
    for cycle in range(ncycle):
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(pv_file_sa, BCL, cycle, SimDet)
        t_systole_sa[cycle] = homo_tptt1[np.argmin(homo_LVV1)] 

    # Load data sa
    data_ecc_sa = np.load(ecc_path_sa)
    n_segments_ecc_sa = len(data_ecc_sa.files)
    segment_strains_ecc_sa = [data_ecc_sa[f'arr_{i}'] for i in range(n_segments_ecc_sa)]
    strain_matrix_ecc_sa = np.vstack(segment_strains_ecc_sa)  # shape: (17, N)
    n_timesteps_ecc_sa = strain_matrix_ecc_sa.shape[1]

    data_ell_sa = np.load(ell_path_sa)
    n_segments_ell_sa = len(data_ell_sa.files)
    segment_strains_ell_sa = [data_ell_sa[f'arr_{i}'] for i in range(n_segments_ell_sa)]
    strain_matrix_ell_sa = np.vstack(segment_strains_ell_sa)  # shape: (17, N)
    n_timesteps_ell_sa = strain_matrix_ell_sa.shape[1]
    frames_per_cycle_sa = n_timesteps_ecc_sa // ncycle    

    #################################################################
    ########### partial posterior block
    #################################################################    
    outdir_pp = "/mnt/home/hagersan/GITLAB/heArt/demo/LBBB-fine/LVelectromechanics-PJ-lbbb954-new"
    ecc_path_pp = os.path.join(outdir_pp, "Ecc_AHA.npz")
    ell_path_pp = os.path.join(outdir_pp, "Ell_AHA.npz") 

    pv_file_pp = os.path.join(outdir_pp+"/BiV_PV.txt")     
    t_systole_pp= np.zeros(ncycle)   
   
    for cycle in range(ncycle):
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(pv_file_pp, BCL, cycle, SimDet)
        t_systole_pp[cycle] = homo_tptt1[np.argmin(homo_LVV1)] 

    # Load data pp
    data_ecc_pp = np.load(ecc_path_pp)
    n_segments_ecc_pp = len(data_ecc_pp.files)
    segment_strains_ecc_pp = [data_ecc_pp[f'arr_{i}'] for i in range(n_segments_ecc_pp)]
    strain_matrix_ecc_pp = np.vstack(segment_strains_ecc_pp)  # shape: (17, N)
    n_timesteps_ecc_pp = strain_matrix_ecc_pp.shape[1]

    data_ell_pp = np.load(ell_path_pp)
    n_segments_ell_pp = len(data_ell_pp.files)
    segment_strains_ell_pp = [data_ell_pp[f'arr_{i}'] for i in range(n_segments_ell_pp)]
    strain_matrix_ell_pp = np.vstack(segment_strains_ell_pp)  # shape: (17, N)
    n_timesteps_ell_pp = strain_matrix_ell_pp.shape[1]
    frames_per_cycle_pp = n_timesteps_ecc_pp // ncycle    

    #################################################################
    ########### lbbb
    #################################################################      
    outdir_lbbb = "/mnt/home/hagersan/GITLAB/heArt/demo/LBBB-fine/LVelectromechanics-PJ-block-test"
    ecc_path_lbbb = os.path.join(outdir_lbbb, "Ecc_AHA.npz")
    ell_path_lbbb = os.path.join(outdir_lbbb, "Ell_AHA.npz") 

    pv_file_lbbb = os.path.join(outdir_lbbb+"/BiV_PV.txt")     
    t_systole_lbbb= np.zeros(ncycle)   
   
    for cycle in range(ncycle):
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(pv_file_lbbb, BCL, cycle, SimDet)
        t_systole_lbbb[cycle] = homo_tptt1[np.argmin(homo_LVV1)]  

    # Load data lbbb
    data_ecc_lbbb = np.load(ecc_path_lbbb)
    n_segments_ecc_lbbb = len(data_ecc_lbbb.files)
    segment_strains_ecc_lbbb = [data_ecc_lbbb[f'arr_{i}'] for i in range(n_segments_ecc_lbbb)]
    strain_matrix_ecc_lbbb = np.vstack(segment_strains_ecc_lbbb)  # shape: (17, N)
    n_timesteps_ecc_lbbb = strain_matrix_ecc_lbbb.shape[1]

    data_ell_lbbb = np.load(ell_path_lbbb)
    n_segments_ell_lbbb = len(data_ell_lbbb.files)
    segment_strains_ell_lbbb = [data_ell_lbbb[f'arr_{i}'] for i in range(n_segments_ell_lbbb)]
    strain_matrix_ell_lbbb = np.vstack(segment_strains_ell_lbbb)  # shape: (17, N)
    n_timesteps_ell_lbbb = strain_matrix_ell_lbbb.shape[1]
    frames_per_cycle_lbbb = n_timesteps_ecc_lbbb // ncycle


    #################################################################
    ######### Calculation

    for cycle in range(ncycle):
        start = cycle * frames_per_cycle_baseline
        end = (cycle + 1) * frames_per_cycle_baseline

        # --- Save strain minima and time indices - baseline ---
        min_ecc_times_baseline = np.argmin(strain_matrix_ecc_baseline[:, start:end], axis=1) + start
        min_ell_times_baseline = np.argmin(strain_matrix_ell_baseline[:, start:end], axis=1) + start

        # convert_frame_index_to_time(min_ecc_times)
        min_ecc_time_ms_baseline = convert_frame_index_to_time(min_ecc_times_baseline, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])
        min_ell_time_ms_baseline = convert_frame_index_to_time(min_ell_times_baseline, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])

        valid_min_ecc_times_ms_baseline = min_ecc_time_ms_baseline[:16]  # segments 1–16 only
        valid_min_ell_times_ms_baseline = min_ell_time_ms_baseline[:16]  # segments 1–16 only

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ecc_baseline = np.full(17, np.nan)
        delta_time_ecc_baseline[:16] = valid_min_ecc_times_ms_baseline - (t_systole_baseline[cycle] if cycle == 0 else t_systole_baseline[cycle] - (cycle - 1) * BCL)

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ell_baseline = np.full(17, np.nan)
        delta_time_ell_baseline[:16] = valid_min_ell_times_ms_baseline - (t_systole_baseline[cycle] if cycle == 0 else t_systole_baseline[cycle] - (cycle - 1) * BCL)        

        # --- Save strain minima and time indices  - lafb ---
        min_ecc_times_lafb = np.argmin(strain_matrix_ecc_lafb[:, start:end], axis=1) + start
        min_ell_times_lafb = np.argmin(strain_matrix_ell_lafb[:, start:end], axis=1) + start

        # convert_frame_index_to_time(min_ecc_times)
        min_ecc_time_ms_lafb = convert_frame_index_to_time(min_ecc_times_lafb, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])
        min_ell_time_ms_lafb = convert_frame_index_to_time(min_ell_times_lafb, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])

        valid_min_ecc_times_ms_lafb = min_ecc_time_ms_lafb[:16]  # segments 1–16 only
        valid_min_ell_times_ms_lafb = min_ell_time_ms_lafb[:16]  # segments 1–16 only

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ecc_lafb = np.full(17, np.nan)
        delta_time_ecc_lafb[:16] = valid_min_ecc_times_ms_lafb - (t_systole_lafb[cycle] if cycle == 0 else t_systole_lafb[cycle] - (cycle - 1) * BCL)

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ell_lafb = np.full(17, np.nan)
        delta_time_ell_lafb[:16] = valid_min_ell_times_ms_lafb - (t_systole_lafb[cycle] if cycle == 0 else t_systole_lafb[cycle] - (cycle - 1) * BCL)   

        # --- Save strain minima and time indices - sa ---
        min_ecc_times_sa = np.argmin(strain_matrix_ecc_sa[:, start:end], axis=1) + start
        min_ell_times_sa = np.argmin(strain_matrix_ell_sa[:, start:end], axis=1) + start

        # convert_frame_index_to_time(min_ecc_times)
        min_ecc_time_ms_sa = convert_frame_index_to_time(min_ecc_times_sa, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])
        min_ell_time_ms_sa = convert_frame_index_to_time(min_ell_times_sa, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])

        valid_min_ecc_times_ms_sa = min_ecc_time_ms_sa[:16]  # segments 1–16 only
        valid_min_ell_times_ms_sa = min_ell_time_ms_sa[:16]  # segments 1–16 only

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ecc_sa = np.full(17, np.nan)
        delta_time_ecc_sa[:16] = valid_min_ecc_times_ms_sa - (t_systole_sa[cycle] if cycle == 0 else t_systole_sa[cycle] - (cycle - 1) * BCL)

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ell_sa = np.full(17, np.nan)
        delta_time_ell_sa[:16] = valid_min_ell_times_ms_sa - (t_systole_sa[cycle] if cycle == 0 else t_systole_sa[cycle] - (cycle - 1) * BCL)        

        # --- Save strain minima and time indices - pp ---
        min_ecc_times_pp = np.argmin(strain_matrix_ecc_pp[:, start:end], axis=1) + start
        min_ell_times_pp = np.argmin(strain_matrix_ell_pp[:, start:end], axis=1) + start

        # convert_frame_index_to_time(min_ecc_times)
        min_ecc_time_ms_pp = convert_frame_index_to_time(min_ecc_times_pp, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])
        min_ell_time_ms_pp = convert_frame_index_to_time(min_ell_times_pp, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])

        valid_min_ecc_times_ms_pp = min_ecc_time_ms_pp[:16]  # segments 1–16 only
        valid_min_ell_times_ms_pp = min_ell_time_ms_pp[:16]  # segments 1–16 only

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ecc_pp = np.full(17, np.nan)
        delta_time_ecc_pp[:16] = valid_min_ecc_times_ms_pp - (t_systole_pp[cycle] if cycle == 0 else t_systole_pp[cycle] - (cycle - 1) * BCL)

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ell_pp = np.full(17, np.nan)
        delta_time_ell_pp[:16] = valid_min_ell_times_ms_pp - (t_systole_pp[cycle] if cycle == 0 else t_systole_pp[cycle] - (cycle - 1) * BCL)        

        # --- Save strain minima and time indices - lbbb---
        min_ecc_times_lbbb = np.argmin(strain_matrix_ecc_lbbb[:, start:end], axis=1) + start
        min_ell_times_lbbb = np.argmin(strain_matrix_ell_lbbb[:, start:end], axis=1) + start

        # convert_frame_index_to_time(min_ecc_times)
        min_ecc_time_ms_lbbb = convert_frame_index_to_time(min_ecc_times_lbbb, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])
        min_ell_time_ms_lbbb = convert_frame_index_to_time(min_ell_times_lbbb, SimDet["dt"], SimDet["writeStep"], SimDet["dt"])

        valid_min_ecc_times_ms_lbbb = min_ecc_time_ms_lbbb[:16]  # segments 1–16 only
        valid_min_ell_times_ms_lbbb = min_ell_time_ms_lbbb[:16]  # segments 1–16 only

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ecc_lbbb = np.full(17, np.nan)
        delta_time_ecc_lbbb[:16] = valid_min_ecc_times_ms_lbbb - (t_systole_lbbb[cycle] if cycle == 0 else t_systole_lbbb[cycle] - (cycle - 1) * BCL)

        # --- Bullseye of time delta between min Ecc and t_systole ---
        delta_time_ell_lbbb = np.full(17, np.nan)
        delta_time_ell_lbbb[:16] = valid_min_ell_times_ms_lbbb - (t_systole_lbbb[cycle] if cycle == 0 else t_systole_lbbb[cycle] - (cycle - 1) * BCL)  

        # --- Bullseye of time delta between min Ecc and t_systole ---
        # Normalize color scale to this delta
        vmin_dt, vmax_dt = np.nanmin(delta_time_ecc_lbbb[:16]), np.nanmax(delta_time_ecc_lbbb[:16])
        norm_dt = mpl.colors.Normalize(vmin=vmin_dt, vmax=vmax_dt)
        cmap_dt = cm.coolwarm

        fig_dt, axes_dt = plt.subplots(2, 5, figsize=(8, 4), subplot_kw=dict(polar=True),gridspec_kw={"wspace": 0.1, "hspace": 0.1})
        axes_dt = axes_dt.reshape(2, 5)  # Ensure shape is consistent

        bullseye_plot(axes_dt[0, 0], data=delta_time_ecc_baseline, seg_bold=[], cmap=cmap_dt, norm=norm_dt)
        bullseye_plot(axes_dt[0, 1], data=delta_time_ecc_lafb, seg_bold=[], cmap=cmap_dt, norm=norm_dt)
        bullseye_plot(axes_dt[0, 2], data=delta_time_ecc_sa, seg_bold=[], cmap=cmap_dt, norm=norm_dt)
        bullseye_plot(axes_dt[0, 3], data=delta_time_ecc_pp, seg_bold=[], cmap=cmap_dt, norm=norm_dt)
        bullseye_plot(axes_dt[0, 4], data=delta_time_ecc_lbbb, seg_bold=[], cmap=cmap_dt, norm=norm_dt)

        bullseye_plot(axes_dt[1, 0], data=delta_time_ell_baseline, seg_bold=[], cmap=cmap_dt, norm=norm_dt)
        bullseye_plot(axes_dt[1, 1], data=delta_time_ell_lafb, seg_bold=[], cmap=cmap_dt, norm=norm_dt)
        bullseye_plot(axes_dt[1, 2], data=delta_time_ell_sa, seg_bold=[], cmap=cmap_dt, norm=norm_dt)
        bullseye_plot(axes_dt[1, 3], data=delta_time_ell_pp, seg_bold=[], cmap=cmap_dt, norm=norm_dt)        
        bullseye_plot(axes_dt[1, 4], data=delta_time_ell_lbbb, seg_bold=[], cmap=cmap_dt, norm=norm_dt)

        # Remove radial labels and ticks for all
        for ax in axes_dt.ravel():
            ax.set_xticks([])
            ax.set_yticks([])        

        # Shared colorbar (optional)
        sm_dt = mpl.cm.ScalarMappable(cmap=cmap_dt, norm=norm_dt)
        sm_dt.set_array([])
        #cbar_dt = fig_dt.colorbar(sm_dt, ax=axes_dt.ravel().tolist(), orientation='horizontal', pad=0.1, shrink=0.85)
        cbar_dt = fig_dt.colorbar(sm_dt, ax=axes_dt.ravel().tolist(),
                          orientation='horizontal',
                          pad=0.08,   # space between subplots and colorbar
                          fraction=0.035,  # relative height of colorbar
                          aspect=30,  # length/height ratio of colorbar
                          shrink=0.65)  # shrink overall size of colorbar
         
        cbar_dt.set_label("ΔT: Min Peak Strain vs Systole [ms]", fontsize=10, labelpad=4)
        # axes_dt.set_title(f"ΔT (Peakstrain min - systole) - Cycle {cycle + 1}", fontsize=14)

        # Save or show
        # plt.tight_layout()
        delta_fig_path = os.path.join(IODet["outputfolder"]+outputdir, f"delta_minPeak_from_systole_cycle{cycle+1}.png")
        plt.savefig(delta_fig_path, dpi=600, bbox_inches='tight')
        plt.close(fig_dt)
        # plt.savefig("all_bullseyes_grid.png", dpi=300)
        # plt.show()



def plot_PV_comparison(IODet, SimDet, cycle=None):
    """Compare pressure-time curves from two different BiV_PV.txt files."""
    BCL = SimDet["HeartBeatLength"]
    block = IODet["block"]
    
    if cycle is None:
        cycle = SimDet["closedloopparam"]["stop_iter"] +1

    plt.figure()
    # Plot first dataset
    for ncycle in range(cycle):
        filename1 = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/BiV_PV.txt")
        time1, LVP1, _, _, _, _ = extract_PV(filename1, BCL, ncycle, SimDet)
        plt.plot(time1, LVP1, label=f"Baseline - Cycle {ncycle}")

    # Plot second dataset
    for ncycle in range(cycle):
        filename2 = os.path.join(IODet["outputfolder"]+IODet["caseID"]+"/BiV_PV.txt")
        time2, LVP2, _, _, _, _ = extract_PV(filename2, BCL, ncycle, SimDet)
        plt.plot(time2, LVP2, linestyle="dashed", label=f"{block} - Cycle {ncycle}")

    # Create output directory if not exists
    hemodynamics_outdirectory = os.path.join(IODet["outputfolder"]+block)
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    plt.legend()
    plt.ylabel("Pressure (mmHg)")
    plt.xlabel("Time [ms]")
    # plt.title("Comparison of Baseline and LBBB Pressure")"_cycle_"+str(cycle)+".png"
    plt.savefig(os.path.join(hemodynamics_outdirectory, "Pressure_Comparison_"+block+"_simulation.png"))
    plt.show()


# def plot_PV_comparison_ind(IODet, SimDet, cycle=None):
#     """Compare pressure-time curves from two different BiV_PV.txt files."""
#     BCL = SimDet["HeartBeatLength"]
#     ncycle = cycle
#     block = IODet["block"]

#     plt.figure()

#     filename1 = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/BiV_PV.txt")
#     time1, LVP1, _, _, _, _ = extract_PV(filename1, BCL, ncycle, SimDet)   
#     plt.plot(time1, LVP1, label=f"Baseline")

#     filename2 = os.path.join(IODet["outputfolder"]+IODet["caseID"]+"/BiV_PV.txt")
#     time2, LVP2, _, _, _, _ = extract_PV(filename2, BCL, ncycle, SimDet)
#     plt.plot(time2, LVP2, linestyle="dashed", label=f"{block}")     

#     # Create output directory if not exists
#     hemodynamics_outdirectory = os.path.join(IODet["outputfolder"]+block)
#     if not os.path.exists(hemodynamics_outdirectory):
#         os.mkdir(hemodynamics_outdirectory)

#     plt.legend()
#     plt.ylabel("Pressure (mmHg)")
#     plt.xlabel("Time (s)")
#     # plt.title(f"Comparison of Baseline and LBBB Pressure {ncycle}")
#     plt.savefig(os.path.join(hemodynamics_outdirectory, "Pressure_Comparison_"+block+"_cycle_"+str(cycle)+".png"))
#     plt.show()    


def plothemodynamics_comparison(IODet, SimDet, cycle=None):
    """Compare PV loops from two different BiV_PV.txt files."""
    BCL = SimDet["HeartBeatLength"]
    block = IODet["block"]    
    
    if cycle is None:
        cycle = SimDet["closedloopparam"]["stop_iter"]+1

    plt.figure()
    # Plot first dataset (solid lines)
    for ncycle in range(cycle):
        filename1 = os.path.join(IODet["outputfolder"]+IODet["directory_baseline"]+"/BiV_PV.txt")
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(filename1, BCL, ncycle, SimDet)
        plt.plot(homo_LVV1, homo_LVP1, label=f"Baseline LV Cycle {ncycle}")

    # Plot second dataset (dashed lines)
    for ncycle in range(cycle):
        filename2 = os.path.join(IODet["outputfolder"]+IODet["caseID"]+"/BiV_PV.txt")
        homo_tptt2, homo_LVP2, homo_LVV2, homo_RVP2, homo_RVV2, homo_Qmv2 = extract_PV(filename2, BCL, ncycle, SimDet)
        plt.plot(homo_LVV2, homo_LVP2, linestyle="dashed", label=f"{block} Cycle {ncycle}")

    # Create output directory if not exists
    hemodynamics_outdirectory = os.path.join(IODet["outputfolder"]+block)
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    plt.legend()
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    # plt.title("Comparison of Baseline and LBBB PV Loops") , 
    plt.savefig(os.path.join(hemodynamics_outdirectory, "PV_Comparison_"+block+"_simulation.png"))
    plt.show()


# def plothemodynamics_comparison_ind(IODet, SimDet, cycle=None):
#     """Compare PV loops from two different BiV_PV.txt files."""
#     BCL = SimDet["HeartBeatLength"]
#     ncycle = cycle
#     block = IODet["block"]     

#     plt.figure()
#     filename1 = os.path.join(IODet["outputfolder"]+"LVelectromechanics-PJ-baseline/BiV_PV.txt")
#     homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(filename1, BCL, ncycle, SimDet)
#     plt.plot(homo_LVV1, homo_LVP1, label=f"Baseline")

#     filename2 = os.path.join(IODet["outputfolder"]+IODet["caseID"]+"/BiV_PV.txt")
#     homo_tptt2, homo_LVP2, homo_LVV2, homo_RVP2, homo_RVV2, homo_Qmv2 = extract_PV(filename2, BCL, ncycle, SimDet)
#     plt.plot(homo_LVV2, homo_LVP2, linestyle="dashed", label=f"{block}")

#     filename3 = os.path.join(IODet["outputfolder"]+"LVelectromechanics-PJ-ilbbb8-new/BiV_PV.txt")
#     homo_tptt3, homo_LVP3, homo_LVV3, homo_RVP3, homo_RVV3, homo_Qmv3 = extract_PV(filename3, BCL, ncycle, SimDet)
#     plt.plot(homo_LVV3, homo_LVP3, linestyle="dashed", label=f"LAFB")    

#     # Create output directory if not exists
#     hemodynamics_outdirectory = os.path.join(IODet["outputfolder"]+block)
#     if not os.path.exists(hemodynamics_outdirectory):
#         os.mkdir(hemodynamics_outdirectory)

#     plt.legend()
#     plt.xlabel("Volume (mL)")
#     plt.ylabel("Pressure (mmHg)")
#     # plt.title(f"Comparison of Baseline and LBBB PV Loops") , 
#     plt.savefig(os.path.join(hemodynamics_outdirectory, "PV_Comparison_"+block+"_cycle_"+str(cycle)+".png"))
#     plt.show()    

# def compute_strain_AHA(IODet, SimDet, LVid=1, cycle=None, R_apex=1.0):
#     isLV = False
#     isBiV = False
#     BCL = SimDet["HeartBeatLength"]
#     block = IODet["block"]  
#     t_vol_min = np.zeros(cycle+1)   

#     for ncycle in range(cycle+1):
#         filename1 = os.path.join(IODet["outputfolder"]+IODet["caseID"]+"/BiV_PV.txt")
#         homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(filename1, BCL, ncycle, SimDet)
#         t_vol_min[ncycle] = homo_tptt1[np.argmin(homo_LVV1)]

#     mesh = df.Mesh()
#     hdf = df.HDF5File(mesh.mpi_comm(),IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5","r",)
#     u_arr = extractdisplacement(IODet, SimDet, cycle=None)

#     if "isLV" in list(SimDet.keys()):
#         isLV = SimDet["isLV"]

#         if(isLV):
#             mesh_params = {
#                 "directory": IODet["directory_me"],
#                 "casename": IODet["casename_me"],
#                 "outputfolder": IODet["outputfolder"],
#                 "foldername": IODet["folderName"],
#                 "isLV": isLV,
#             }

#             Mesh_obj = lv_mechanics_mesh(mesh_params, SimDet)

#             try:
#                 eC0 = Mesh_obj.eC0
#                 eL0 = Mesh_obj.eL0
#                 eR0 = Mesh_obj.eR0

#             except AttributeError:
#                 fiber_angle_param = {   "mesh": Mesh_obj.mesh,
#                                         "facetboundaries": Mesh_obj.facetboundaries,
#                                         "LV_fiber_angle": [0.01, -0.01],
#                                         "LV_sheet_angle": [0.1, -0.1],
#                                         "minztol": Mesh_obj.mesh.hmax()/2.0, # Coarse mesh
#                                         "isrotatept": False,
#                                         "isreturn": True,
#                                         "outfilename": IODet["casename_me"],
#                                         "outdirectory": IODet["outputfolder"] + IODet["caseID"] + IODet["folderName"],
#                                         "baseid": SimDet["topid"],
#                                         "epiid": SimDet["epiid"],
#                                         "lvid": SimDet["LVendoid"],
#                                         "degree": SimDet["GiccioneParams"]["deg"]}

#                 eC0, eL0, eR0  = vtk_py.addLVfiber_LDRB(fiber_angle_param)

#     deg = SimDet["GiccioneParams"]["deg"]
#     # Normalize directions
#     eC0_normalized = eC0 / df.sqrt(df.inner(eC0, eC0))
#     eL0_normalized = eL0 / df.sqrt(df.inner(eL0, eL0))
#     eR0_normalized = eR0 / df.sqrt(df.inner(eR0, eR0))

#     # === Tag AHA Segments ===
#     def tag_AHA_segments(mesh, apex_axis="z", R_apex=2.0, reference_point=SimDet["HIS"]):
#         mesh.init()
#         cell_tags = df.MeshFunction("size_t", mesh, mesh.topology().dim())
#         cell_tags.set_all(0)

#         coords = mesh.coordinates()
#         z_vals = coords[:, 2]  # assuming z is apex-base
#         z_min, z_max = z_vals.min(), z_vals.max()
#         h = z_max - z_min
#         basal_th = z_min + h / 3
#         mid_th = z_min + 2 * h / 3

#         if isinstance(reference_point, list) and len(reference_point) == 1 and isinstance(reference_point[0], (list, tuple, np.ndarray)):
#             reference_point = reference_point[0]        

#         # Rotation: find angle to reference point (e.g., lateral wall)
#         if reference_point is not None:
#             x_ref, y_ref = reference_point[0], reference_point[1]
#             theta_ref = np.arctan2(y_ref, x_ref)
#         else:
#             theta_ref = 0.0

#         for cell in df.cells(mesh):
#             mp = cell.midpoint()
#             x, y, z = mp.x(), mp.y(), mp.z()

#             # Rotate angle
#             theta = np.arctan2(y, x) - theta_ref
#             theta = np.degrees(theta) % 360  # Map to [0, 360)

#             # Segment by z
#             if z <= basal_th:
#                 level = "basal"
#             elif z <= mid_th:
#                 level = "mid"
#             else:
#                 level = "apical"

#             # AHA segment ID based on rotated angle
#             if level == "basal":
#                 tag = int(theta // 60) + 1     # 1–6
#             elif level == "mid":
#                 tag = int(theta // 60) + 7     # 7–12
#             elif level == "apical":
#                 tag = int(theta // 90) + 13    # 13–16

#             # Apex (segment 17)
#             if np.linalg.norm([x, y, z - z_max]) < R_apex:
#                 tag = 17

#             cell_tags[cell.index()] = tag

#         return cell_tags

#     aha_segments = tag_AHA_segments(Mesh_obj.mesh)
#     dx_aha = df.Measure("dx", domain=Mesh_obj.mesh, subdomain_data=aha_segments)

#     if SimDet["Mechanics Discretization"] is "P1P1":
#         var_deg = 1
#     else:
#         var_deg = 2

#     # === Setup spaces and strain computation ===
#     udisp = df.Function(df.VectorFunctionSpace(Mesh_obj.mesh, "CG", var_deg))
#     udisp.vector()[:] = u_arr[0].vector().get_local()[:]

#     GuccioneParams = SimDet["GiccioneParams"]
#     params = {  "mesh": Mesh_obj.mesh,
#                 "displacement_variable": udisp,
#                 "material model": SimDet["GiccioneParams"]["Passive model"],
#                 "material params": SimDet["GiccioneParams"]["Passive params"],
#                 "incompressible": SimDet["GiccioneParams"]["incompressible"],
#                 "growth_tensor": None,}

#     uflforms = Forms(params)
#     Fref = df.project(uflforms.Fmat(), df.TensorFunctionSpace(Mesh_obj.mesh, "DG", 0))

#     # === Storage ===
#     n_seg = 17
#     Ecc_arr = [[] for _ in range(n_seg)]
#     Ell_arr = [[] for _ in range(n_seg)]
#     Err_arr = [[] for _ in range(n_seg)]

#     # === Loop over time ===
#     for u_t in u_arr:
#         udisp.vector()[:] = u_t.vector().get_local()[:]
#         Fmat = uflforms.Fmat()
#         F = Fmat * df.inv(Fref)
#         Cmat = F.T * F

#         Ccc = df.inner(eC0_normalized, Cmat * eC0_normalized)
#         Cll = df.inner(eL0_normalized, Cmat * eL0_normalized)  
#         Crr = df.inner(eR0_normalized, Cmat * eR0_normalized)    

#         Ecc = 0.5 * (1 - 1 / Ccc)      
#         Ell = 0.5 * (1 - 1 / Cll)        
#         Err = 0.5 * (1 - 1 / Crr)

#         volumes = [df.assemble(df.Constant(1.0) * dx_aha(seg)) for seg in range(1, 18)]

#         for seg in range(1, n_seg + 1):
#             vol = volumes[seg - 1]

#             if vol > 0:
#                 Ecc_arr[seg - 1].append(df.assemble(Ecc * dx_aha(seg)) / vol)
#                 Ell_arr[seg - 1].append(df.assemble(Ell * dx_aha(seg)) / vol)
#                 Err_arr[seg - 1].append(df.assemble(Err * dx_aha(seg)) / vol)
#             else:
#                 Ecc_arr[seg - 1].append(np.nan)
#                 Ell_arr[seg - 1].append(np.nan)
#                 Err_arr[seg - 1].append(np.nan)

#     # === Save output ===
#     outdir = os.path.join(IODet["outputfolder"], IODet["caseID"])
#     np.savez(os.path.join(outdir, "Ecc_AHA.npz"), *Ecc_arr)
#     np.savez(os.path.join(outdir, "Ell_AHA.npz"), *Ell_arr)
#     np.savez(os.path.join(outdir, "Err_AHA.npz"), *Err_arr)

#     for name, arr in zip(["Ecc", "Ell", "Err"], [Ecc_arr, Ell_arr, Err_arr]):
#         plt.figure(figsize=(10, 5))
#         for seg in range(17):
#             plt.plot(arr[seg], label=f"Seg {seg + 1}")
#         plt.title(f"{name} per AHA segment")
#         plt.xlabel("Time step")
#         plt.ylabel(name)
#         plt.legend(ncol=3, fontsize=8)
#         plt.tight_layout()
#         plt.savefig(os.path.join(outdir, f"{name}_AHA.png"))
#         plt.close()

#     # === Save output ===
#     outdir = os.path.join(IODet["outputfolder"], IODet["caseID"])
#     np.savez(os.path.join(outdir, "Ecc_AHA.npz"), *Ecc_arr)
#     np.savez(os.path.join(outdir, "Ell_AHA.npz"), *Ell_arr)
#     np.savez(os.path.join(outdir, "Err_AHA.npz"), *Err_arr)

#     for name, arr in zip(["Ecc", "Ell", "Err"], [Ecc_arr, Ell_arr, Err_arr]):
#         plt.figure(figsize=(10, 5))
#         for seg in range(n_seg):
#             plt.plot(arr[seg], label=f"Seg {seg+1}")
#         plt.title(f"{name} per AHA segment")
#         plt.xlabel("Time step")
#         plt.ylabel(name)
#         plt.legend(ncol=3, fontsize=8)
#         plt.tight_layout()
#         plt.savefig(os.path.join(outdir, f"{name}_AHA.png"))
#         plt.close()

#     pdb.set_trace()
#     # === Compute SDI from Ecc peak time ===
#     time_vec = homo_tptt1[:len(Ecc_arr[0])]
#     t_min_vol = t_vol_min[0]

#     ecc_peak_times = []
#     for seg in range(17):
#         ecc = np.array(Ecc_arr[seg])
#         if np.all(np.isnan(ecc)):
#             ecc_peak_times.append(np.nan)
#         else:
#             # peak_idx = np.argmax(ecc)  # Or np.argmin(ecc) depending on shortening
#             peak_idx = np.argmin(ecc)
#             ecc_peak_times.append(time_vec[peak_idx])

#     ecc_deviation = [abs(t - t_min_vol) for t in ecc_peak_times if not np.isnan(t)]
#     ecc_sdi = np.std(ecc_deviation)

#     with open(os.path.join(outdir, "ecc_sdi.txt"), "w") as f:
#         f.write(f"SDI (Ecc): {ecc_sdi:.4f} ms\n")
#         f.write("Per-segment deviation (ms):\n")
#         for i, d in enumerate(ecc_deviation):
#             f.write(f"Segment {i+1}: {d:.4f}\n")

#     plt.figure()
#     plt.bar(range(1, 18), ecc_deviation)
#     plt.axhline(ecc_sdi, color='r', linestyle='--', label='SDI')
#     plt.xlabel("AHA Segment")
#     plt.ylabel("|t_peak_strain - t_min_vol| (ms)")
#     plt.title("Segment-wise strain timing deviation")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(outdir, "ecc_sdi_barplot.png"))
#     plt.close()

#     # === Compute SDI from Ell peak time ===
#     time_vec = homo_tptt1[:len(Ell_arr[0])]
#     t_min_vol = t_vol_min[0]

#     ell_peak_times = []
#     for seg in range(17):
#         ell = np.array(Ell_arr[seg])
#         if np.all(np.isnan(ell)):
#             ell_peak_times.append(np.nan)
#         else:
#             # peak_idx = np.argmax(ell)  # Or np.argmin(ell) depending on shortening
#             peak_idx = np.argmin(ell)
#             ell_peak_times.append(time_vec[peak_idx])

#     ell_deviation = [abs(t - t_min_vol) for t in ell_peak_times if not np.isnan(t)]
#     ell_sdi = np.std(ell_deviation)

#     with open(os.path.join(outdir, "ell_sdi.txt"), "w") as f:
#         f.write(f"SDI (Ell): {ell_sdi:.4f} ms\n")
#         f.write("Per-segment deviation (ms):\n")
#         for i, d in enumerate(ell_deviation):
#             f.write(f"Segment {i+1}: {d:.4f}\n")

#     plt.figure()
#     plt.bar(range(1, 18), ell_deviation)
#     plt.axhline(ell_sdi, color='r', linestyle='--', label='SDI')
#     plt.xlabel("AHA Segment")
#     plt.ylabel("|t_peak_strain - t_min_vol| (ms)")
#     plt.title("Segment-wise strain timing deviation")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(outdir, "ell_sdi_barplot.png"))
#     plt.close()            

def assign_aha_tags(mesh, ref_point):
    mesh.init()
    dim = mesh.topology().dim()
    aha_tags = df.MeshFunction("size_t", mesh, dim)
    aha_tags.set_all(0)

    coords = mesh.coordinates()
    z_min, z_max = np.min(coords[:, 2]), np.max(coords[:, 2])
    h = z_max - z_min
    basal_th = z_min + h / 3
    mid_th = z_min + 2 * h / 3
    R_apex = 0.05 * h  # 5% of total LV height

    # pdb.set_trace()
    ref_point = ref_point[0]
    x_ref, y_ref = ref_point[:2]
    theta_ref_deg = np.degrees(np.arctan2(y_ref, x_ref)) 

    for cell in df.cells(mesh):
        mp = cell.midpoint()
        x, y, z = mp.x(), mp.y(), mp.z()

        theta = (np.degrees(np.arctan2(y, x)) - theta_ref_deg) % 360

        if z <= basal_th:
            tag = int(theta // 60) + 1
        elif z <= mid_th:
            tag = int(theta // 60) + 7
        else:
            tag = int(theta // 90) + 13

        # if np.linalg.norm([x, y, z - z_max]) < R_apex:
        #     tag = 17

        dz = z - z_max
        dr = np.linalg.norm([x - x_ref, y - y_ref])
        if dz**2 + dr**2 < R_apex**2 and z > mid_th:
            tag = 17        

        aha_tags[cell] = tag

    return aha_tags
 

def peak_strain_analysis(u_arr, mesh, udisp, uflforms, Fref, eC0_norm, aha_tags):
    V_dg0 = df.FunctionSpace(mesh, "DG", 0)
    Ecc_proj = df.Function(V_dg0)
    
    min_Ecc_AHA = np.full(18, np.inf)
    min_Ecc_time = np.full(18, -1, dtype=int)

    tag_array = aha_tags.array()
    # num_cells = mesh.topology.index_map(mesh.topology.dim()).size_local
    num_cells = mesh.num_cells()

    for t_idx, u in enumerate(tqdm(u_arr, desc="Computing peak Ecc")):
        udisp.vector()[:] = u.vector().get_local()
        F = uflforms.Fmat() * df.inv(Fref)
        C = F.T * F
        Ecc_expr = 0.5 * (1 - 1 / df.inner(eC0_norm, C * eC0_norm))

        # Ecc_proj.interpolate(Ecc_expr)
        Ecc_proj.assign(df.project(Ecc_expr, Ecc_proj.function_space()))
        cell_values = Ecc_proj.vector().get_local()

        for seg_id in range(1, 18):
            segment_cells = np.where(tag_array[:num_cells] == seg_id)[0]
            if segment_cells.size == 0:
                continue
            segment_vals = cell_values[segment_cells]
            min_val = np.min(segment_vals)
            if min_val < min_Ecc_AHA[seg_id]:
                min_Ecc_AHA[seg_id] = min_val
                min_Ecc_time[seg_id] = t_idx

    return {
        seg: min_Ecc_AHA[seg] for seg in range(1, 18)
    }, {
        seg: min_Ecc_time[seg] for seg in range(1, 18)
    }

def compute_strain_AHA(IODet, SimDet, LVid=1, RVid=2, cycle=None, ref_point=None, aha_tags=None): #serial

    outdir = os.path.join(IODet["outputfolder"], IODet["caseID"])

    # --- Mesh and displacement ---
    mesh = df.Mesh()
    hdf = df.HDF5File(mesh.mpi_comm(), os.path.join(IODet["outputfolder"], IODet["caseID"], "Data.h5"), "r")
    hdf.read(mesh, "ME/mesh", False)
    ncycle = cycle
    cycle = cycle + 1

    u_arr = extractdisplacement(IODet, SimDet, cycle)

    n_frames = len(u_arr)
    frames_per_cycle = n_frames // (cycle) # = 320 if n_frames = 640

    start = ncycle * frames_per_cycle
    end = (cycle) * frames_per_cycle    
    
    # pdb.set_trace()
    # Extract second cycle
    u_arr_cycle1 = u_arr[start:end]    

    # --- Mechanics mesh setup ---
    mesh_params = {
        "directory": IODet["directory_me"],
        "casename": IODet["casename_me"],
        "outputfolder": IODet["outputfolder"],
        "foldername": IODet["folderName"],
        "isLV": True,
    }
    Mesh_obj = lv_mechanics_mesh(mesh_params, SimDet)
    mesh = Mesh_obj.mesh
    var_deg = 1 if SimDet["Mechanics Discretization"] == "P1P1" else 2
    udisp = df.Function(df.VectorFunctionSpace(mesh, "CG", var_deg))

    try:
        eC0 = Mesh_obj.eC0
        eL0 = Mesh_obj.eL0
        eR0 = Mesh_obj.eR0

    except AttributeError:
        fiber_angle_param = {   "mesh": Mesh_obj.mesh,
                                "facetboundaries": Mesh_obj.facetboundaries,
                                "LV_fiber_angle": [0.01, -0.01],
                                "LV_sheet_angle": [0.1, -0.1],
                                "minztol": Mesh_obj.mesh.hmax()/2.0, # Coarse mesh
                                "isrotatept": False,
                                "isreturn": True,
                                "outfilename": IODet["casename_me"],
                                "outdirectory": IODet["outputfolder"] + IODet["caseID"] + IODet["folderName"],
                                "baseid": SimDet["topid"],
                                "epiid": SimDet["epiid"],
                                "lvid": SimDet["LVendoid"],
                                "degree": SimDet["GiccioneParams"]["deg"]}

        eC0, eL0, eR0  = vtk_py.addLVfiber_LDRB(fiber_angle_param)

    eC0_norm = eC0 / df.sqrt(df.inner(eC0, eC0))
    eL0_norm = eL0 / df.sqrt(df.inner(eL0, eL0))
    eR0_norm = eR0 / df.sqrt(df.inner(eR0, eR0))

    GuccioneParams = SimDet["GiccioneParams"]
    uflforms = Forms({
        "mesh": mesh,
        "displacement_variable": udisp,
        "material model": GuccioneParams["Passive model"],
        "material params": GuccioneParams["Passive params"],
        "incompressible": GuccioneParams["incompressible"],
        "growth_tensor": None,
    })

    Fref = df.project(uflforms.Fmat(), df.TensorFunctionSpace(mesh, "DG", 0))

    # --- AHA tag assignment ---
    if aha_tags is None:
        if ref_point is None:
            ref_point = SimDet["HIS"]
        aha_tags = assign_aha_tags(mesh, ref_point)

    aha_counts = collections.Counter()

    for cell in df.cells(mesh):
        tag = aha_tags[cell]
        aha_counts[tag] += 1

    # Print the counts
    print("AHA Segment Element Counts:")
    for tag in sorted(aha_counts):
        print(f"  Segment {tag}: {aha_counts[tag]} elements")

    # --- Strain computation ---
    min_Ecc_AHA1, min_Ecc_time1 = peak_strain_analysis(u_arr_cycle1, mesh, udisp, uflforms, Fref, eC0_norm, aha_tags)
    min_Ell_AHA1, min_Ell_time1 = peak_strain_analysis(u_arr_cycle1, mesh, udisp, uflforms, Fref, eL0_norm, aha_tags)
   
    min_Ecc_time1_ms = convert_frame_index_to_time(min_Ecc_time1)
    min_Ell_time1_ms = convert_frame_index_to_time(min_Ell_time1)

    # pdb.set_trace

    # Filter out invalid values (e.g., -1)
    valid_Ecc_times = [t for t in min_Ecc_time1_ms.values() if t != -1]
    valid_Ell_times = [t for t in min_Ell_time1_ms.values() if t != -1]

    # Convert to NumPy array for convenience
    Ecc_times_array = np.array(valid_Ecc_times)    
    Ell_times_array = np.array(valid_Ell_times)  

    # --- Compute SDI ---
    # pdb.set_trace()
    t_systole= np.zeros(cycle)   
    BCL = SimDet["HeartBeatLength"]    

    for ncycle in range(cycle):
        filename1 = os.path.join(IODet["outputfolder"]+IODet["caseID"]+"/BiV_PV.txt")
        homo_tptt1, homo_LVP1, homo_LVV1, homo_RVP1, homo_RVV1, homo_Qmv1 = extract_PV(filename1, BCL, ncycle, SimDet)
        t_systole[ncycle] = homo_tptt1[np.argmin(homo_LVV1)]

    if t_systole is None or len(t_systole) < cycle:
        raise ValueError("SimDet must include 't_min_vol' with at least two entries")

    # # Absolute differences from the reference
    # Ecc_differences = np.abs(Ecc_times_array - (t_systole[1]-800))
    # Ell_differences = np.abs(Ell_times_array - (t_systole[1]-800))

    SDI_Ecc = np.std(Ecc_times_array - (t_systole[cycle-1]-BCL)) / BCL * 100
    SDI_Ell = np.std(Ell_times_array - (t_systole[cycle-1]-BCL)) / BCL * 100    

    print(f"Ecc SDI: {SDI_Ecc:.4f}")
    print(f"Ell SDI: {SDI_Ell:.4f}")

    with open(outdir+"/SDI.txt", "w") as f:
        f.write("SDI\n")
        f.write(f"Ecc SDI:\t{SDI_Ecc:.4f}[%]\n")  
        f.write(f"Ell SDI:\t{SDI_Ell:.4f}[%]\n")  

    # Convert dictionary to ordered list of 17 values
    Ecc_data_dict = min_Ecc_AHA1
    Ecc_data_list = [Ecc_data_dict.get(i, np.nan) for i in range(1, 18)]  # 1-based AHA segments

    Ecc_min_val = np.nanmin(Ecc_data_list)
    Ecc_max_val = np.nanmax(Ecc_data_list)

    # Optional: define normalization and segments to highlight
    Ecc_norm = mpl.colors.Normalize(vmin=min(Ecc_data_list), vmax=max(Ecc_data_list))  # scale colors
    Ecc_cmap = plt.cm.viridis

    # Create figure and polar axes
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Call the plotting function
    bullseye_plot(ax, Ecc_data_list, seg_bold=[], cmap=Ecc_cmap, norm=Ecc_norm)

    # Add colorbar (legend)
    # Create dummy mappable for colorbar
    ticks = np.linspace(Ecc_min_val, Ecc_max_val, 5)
    sm = mpl.cm.ScalarMappable(cmap=Ecc_cmap, norm=Ecc_norm)
    sm.set_array([])  # Required for compatibility
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
    cbar.set_label("Peak Ecc Strain", fontsize=12)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.3f}" for t in ticks])  # formatted ticks


    # Add title and save
    plt.title("Peak concentric Strain - Cycle 2", fontsize=14)
    plt.savefig(outdir+"/peak_Ecc_bullseye.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("Saved:peak_Ecc_bullseye.png")

    # Convert dictionary to ordered list of 17 values
    Ell_data_dict = min_Ell_AHA1
    Ell_data_list = [Ell_data_dict.get(i, np.nan) for i in range(1, 18)]  # 1-based AHA segments

    Ell_min_val = np.nanmin(Ell_data_list)
    Ell_max_val = np.nanmax(Ell_data_list)

    # Optional: define normalization and segments to highlight
    Ell_norm = mpl.colors.Normalize(vmin=min(Ell_data_list), vmax=max(Ell_data_list))  # scale colors
    Ell_cmap = plt.cm.viridis

    # Create figure and polar axes
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Call the plotting function
    bullseye_plot(ax, Ell_data_list, seg_bold=[], cmap=Ell_cmap, norm=Ell_norm)

    # Add colorbar (legend)
    # Create dummy mappable for colorbar
    ticks = np.linspace(Ell_min_val, Ell_max_val, 5)
    sm = mpl.cm.ScalarMappable(cmap=Ell_cmap, norm=Ell_norm)
    sm.set_array([])  # Required for compatibility
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
    cbar.set_label("Peak longitudinal Strain", fontsize=12)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.3f}" for t in ticks])  # formatted ticks

    # Add title and save
    plt.title("Peak longitudinal Strain - Cycle 2", fontsize=14)
    plt.savefig(outdir+"/peak_Ell_bullseye.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("Saved:peak_Ell_bullseye.png")    

    # Prepare time and strain data lists (1–17)
    Ecc_strain_dict = min_Ecc_AHA1
    Ecc_time_dict = min_Ecc_time1_ms

    segments = list(range(1, 18))
    Ecc_strain_list = [Ecc_strain_dict.get(i, np.nan) for i in segments]
    Ecc_time_list = [Ecc_time_dict.get(i, np.nan) for i in segments]

    # ----- Save data to TXT -----
    output_txt = "/peak_Ecc_data.txt"
    with open(outdir+output_txt, "w") as f:
        f.write("Segment\tPeak_Ecc\tTime_to_Peak_Ecc (ms)\n")
        for i, strain, time in zip(segments, Ecc_strain_list, Ecc_time_list):
            f.write(f"{i}\t{strain:.4f}\t{time:.1f}\n")

    print(f"Saved: {output_txt}")

    # ----- Plot: Time to peak Ecc -----
    time_min = np.nanmin(Ecc_time_list)
    time_max = np.nanmax(Ecc_time_list)
    time_norm = mpl.colors.Normalize(vmin=time_min, vmax=time_max)
    cmap_time = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    bullseye_plot(ax, Ecc_time_list, seg_bold=[], cmap=cmap_time, norm=time_norm)

    # Colorbar with time scale
    ticks_time = np.linspace(time_min, time_max, 5)
    sm_time = mpl.cm.ScalarMappable(cmap=cmap_time, norm=time_norm)
    sm_time.set_array([])
    cbar = plt.colorbar(sm_time, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
    cbar.set_label("Time to Peak concentric Strain (ms)", fontsize=12)
    cbar.set_ticks(ticks_time)
    cbar.set_ticklabels([f"{t:.0f}" for t in ticks_time])

    plt.title("Time to Peak concentric Strain - Cycle 2", fontsize=14)
    plt.savefig(outdir+"/time_to_peak_Ecc_bullseye.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("Saved: cycle2_time_to_peak_Ecc_bullseye.png")


    # Prepare time and strain data lists (1–17)
    Ell_strain_dict = min_Ell_AHA1
    Ell_time_dict = min_Ell_time1_ms

    segments = list(range(1, 18))
    Ell_strain_list = [Ell_strain_dict.get(i, np.nan) for i in segments]
    Ell_time_list = [Ell_time_dict.get(i, np.nan) for i in segments]

    # ----- Save data to TXT -----
    output_txt = "/peak_Ell_data.txt"
    with open(outdir+output_txt, "w") as f:
        f.write("Segment\tPeak_Ell\tTime_to_Peak_Ell (ms)\n")
        for i, strain, time in zip(segments, Ell_strain_list, Ell_time_list):
            f.write(f"{i}\t{strain:.4f}\t{time:.1f}\n")

    print(f"Saved: {output_txt}")

    # ----- Plot: Time to peak Ell -----
    time_min = np.nanmin(Ell_time_list)
    time_max = np.nanmax(Ell_time_list)
    time_norm = mpl.colors.Normalize(vmin=time_min, vmax=time_max)
    cmap_time = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    bullseye_plot(ax, Ell_time_list, seg_bold=[], cmap=cmap_time, norm=time_norm)

    # Colorbar with time scale
    ticks_time = np.linspace(time_min, time_max, 5)
    sm_time = mpl.cm.ScalarMappable(cmap=cmap_time, norm=time_norm)
    sm_time.set_array([])
    cbar = plt.colorbar(sm_time, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
    cbar.set_label("Time to Peak longitudinal Strain (ms)", fontsize=12)
    cbar.set_ticks(ticks_time)
    cbar.set_ticklabels([f"{t:.0f}" for t in ticks_time])

    plt.title("Time to Peak longitudinal Strain - Cycle 2", fontsize=14)
    plt.savefig(outdir+"/time_to_peak_Ell_bullseye.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("Saved: cycle2_time_to_peak_Ell_bullseye.png")    

    # plot_bullseye_aha(min_Ecc_AHA1, title="Peak Eccentric Strain", save_path="cycle2_peak_Ecc_strain.png")
    return min_Ecc_AHA1, min_Ecc_time1, min_Ecc_time1_ms, SDI_Ecc, min_Ell_AHA1, min_Ell_time1, min_Ell_time1_ms, SDI_Ell


def convert_frame_index_to_time(min_Ecc_time, dt=0.5, step_interval=5, t0=0.5):
    return {seg: (t0 + dt * step_interval * t_idx) if t_idx >= 0 else -1
            for seg, t_idx in min_Ecc_time.items()}


def bullseye_plot(ax, data, seg_bold=None, cmap="viridis", norm=None):
    """
    Bullseye representation for the left ventricle.

    Parameters
    ----------
    ax : Axes
    data : list[float]
        The intensity values for each of the 17 segments.
    seg_bold : list[int], optional
        A list with the segments to highlight.
    cmap : colormap, default: "viridis"
        Colormap for the data.
    norm : Normalize or None, optional
        Normalizer for the data.

    Notes
    -----
    This function creates the 17 segment model for the left ventricle according
    to the American Heart Association (AHA) [1]_

    References
    ----------
    .. [1] M. D. Cerqueira, N. J. Weissman, V. Dilsizian, A. K. Jacobs,
        S. Kaul, W. K. Laskey, D. J. Pennell, J. A. Rumberger, T. Ryan,
        and M. S. Verani, "Standardized myocardial segmentation and
        nomenclature for tomographic imaging of the heart",
        Circulation, vol. 105, no. 4, pp. 539-542, 2002.
    """

    data = np.ravel(data)
    if seg_bold is None:
        seg_bold = []
    if norm is None:
        norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())

    r = np.linspace(0.2, 1, 4)

    ax.set(ylim=[0, 1], xticklabels=[], yticklabels=[])
    ax.grid(False)  # Remove grid

    # Fill segments 1-6, 7-12, 13-16.
    for start, stop, r_in, r_out in [
            (0, 6, r[2], r[3]),
            (6, 12, r[1], r[2]),
            (12, 16, r[0], r[1]),
            (16, 17, 0, r[0]),
    ]:
        n = stop - start
        dtheta = 2*np.pi / n
        ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
               color=cmap(norm(data[start:stop])))

    # Now, draw the segment borders.  In order for the outer bold borders not
    # to be covered by inner segments, the borders are all drawn separately
    # after the segments have all been filled.  We also disable clipping, which
    # would otherwise affect the outermost segment edges.
    # Draw edges of segments 1-6, 7-12, 13-16.
    for start, stop, r_in, r_out in [
            (0, 6, r[2], r[3]),
            (6, 12, r[1], r[2]),
            (12, 16, r[0], r[1]),
    ]:
        n = stop - start
        dtheta = 2*np.pi / n
        ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
               clip_on=False, color="none", edgecolor="k", linewidth=[
                   4 if i + 1 in seg_bold else 2 for i in range(start, stop)])
    # Draw edge of segment 17 -- here; the edge needs to be drawn differently,
    # using plot().
    ax.plot(np.linspace(0, 2*np.pi), np.linspace(r[0], r[0]), "k",
            linewidth=(4 if 17 in seg_bold else 2))

    # # Add segment numbers for reference
    # seg_num = 1
    # label_config = [
    #     (6, r[2] + (r[3] - r[2]) / 2),   # outer ring: seg 1–6
    #     (6, r[1] + (r[2] - r[1]) / 2),   # mid ring: seg 7–12
    #     (4, r[0] + (r[1] - r[0]) / 2),   # inner ring: seg 13–16
    #     (1, r[0] / 2)                    # center: seg 17
    # ]

    # for ring_idx, (n_segments, radius) in enumerate(label_config):
    #     dtheta = 2 * np.pi / n_segments
    #     offset = np.pi / 2  # to start from top

    #     for i in range(n_segments):
    #         angle = offset + i * dtheta
    #         ax.text(angle, radius, str(seg_num),
    #                 ha='center', va='center', fontsize=10, color='black')
    #         seg_num += 1    