import sys, pdb, vtk
import dolfin
from dolfin import *
import numpy as np

sys.setrecursionlimit(5000)  # Increase to a higher value
sys.path.append("/mnt/Research/heArt")
sys.path.append("/mnt/Research")
sys.path.append("/mnt/Output")
sys.path.append("/home/fenicstools")

import vtk_py3

from heArt.src.sim_protocols.run_BiV_ClosedLoop_pctrl_test2 import (run_BiV_ClosedLoop as run_BiV_ClosedLoop,)
from heArt.src.postprocessing.postprocessdata2 import postprocessdata as postprocessdata
from heArt.src.postprocessing.postprocessdata2 import dumpvtk as dumpvtk
from heArt.src.postprocessing.postprocessdata2 import compute_strain as compute_strain
from heArt.src.postprocessing.postprocessdata2 import compute_strain_split as compute_strain_split
from heArt.src.postprocessing.postprocessdata2 import compute_strain_AHA as compute_strain_AHA
from heArt.src.postprocessing.postprocessdata2 import compute_strain_AHA_profile as compute_strain_AHA_profile
from heArt.src.postprocessing.postprocessdata2 import compute_activation as compute_activation
from heArt.src.postprocessing.postprocessdata2 import plothemodynamics as plothemodynamics
from heArt.src.postprocessing.postprocessdata2 import plotpressure as plotpressure
from heArt.src.postprocessing.postprocessdata2 import plot_PV_comparison as plot_PV_comparison
from heArt.src.postprocessing.postprocessdata2 import plot_PV_comparison_ind as plot_PV_comparison_ind
from heArt.src.postprocessing.postprocessdata2 import plothemodynamics_comparison as plothemodynamics_comparison
from heArt.src.postprocessing.postprocessdata2 import plothemodynamics_comparison_ind as plothemodynamics_comparison_ind
from heArt.src.postprocessing.postprocessdata2 import plot_min_bullseye_per_cycle as plot_min_bullseye_per_cycle
from heArt.src.postprocessing.postprocessdata2 import check_AHA_plot as check_AHA_plot
from heArt.src.postprocessing.postprocessdata2 import strain_plot_AHA as strain_plot_AHA
#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
IODetails = {
    "casename_me": "ellipsoidal_baselinegeo_medium3",
    "casename_ep": "ellipsoidal_baselinegeo_fine1",
    "directory_me": "../LVMesh/",
    "directory_ep": "../LVMesh/",
    "directory_pj": "../LV_pj_mesh_refined/8/", # LAFB:8 / LSAFB:398/ LAFB:954
    "casename_pj": "PJmarked",
    "casename_pj_marked": "PJmarked",     
    "outputfolder": "./",    
    "folderName": "/",   
    "caseID": "PJ",  
    "block":"block", # for postprocess comparison    
    "isLV": True,
}

contRactility = 1000e3

GuccioneParams = {
    "ParamsSpecified": True,
    "Passive model": {"Name": "Guccione"},
    "Passive params": {
        "Cparam": Constant(50.0),
        "bff": Constant(29.0),
        "bfx": Constant(13.3),
        "bxx": Constant(26.6),
    },
    "Active model": {"Name": "Time-varying"},
    "Active params": {
        "tau": 25,
        "t_trans": 300,
        "B": 4.75,
        "t0": 275,
        "l0": 1.58,
        "Tmax": Constant(contRactility),
        "Ca0": 4.35,
        "Ca0max": 4.35,
        "lr": 1.85,
    },
    "HomogenousActivation": False,
    "deg": 4,
    "Kappa": 1e6,
    "incompressible": True,
}
Circparam = {
    "Ees_la": 10,
    "A_la": 2.67,
    "B_la": 0.019,
    "V0_la": 10,
    "Tmax_la": 120,
    "tau_la": 25,
    "tdelay_la": 160,
    "Csa": 0.0052,
    "Cad": 0.013,
    "Csv": 0.28,
    "Vsa0": 360,
    "Vad0": 40,
    "Vsv0": 3370.0,
    "Rav": 2000,
    "Rsv": 100.0,
    "Rsa": 58000,
    "Rad": 106000,
    "Rmv": 2000,
    "V_sa": 407.9870929796549, 
    "V_ad": 139.88354730982294,
    "V_sv": 3800.6771443568937,
    "V_LA": 193.99555092431984,
    "V_LV": 98.40525741977021, 
    "stop_iter": 1,
    "issoftplus": False,    
}

SimDetails = {
    "diaplacementInfo_ref": False,
    # Time discretization:
    "HeartBeatLength": 800.0,
    "dt": 0.5,
    "writeStep": 5.0,
    "DTI_EP": False,
    "DTI_ME": False,
    # Circulatory system:
    "Isclosed": True,
    "closedloopparam": Circparam,   
    "EDP": 12.342740722563716,    
    "nLoadSteps": 25,     
    # Boundary Condition
    "springbc": True,
    "Mechanics Discretization": "P1P1",
    "spring_atbase": True,
    "topid": 4,
    "LVendoid": 2,
    "RVendoid": 0,
    "epiid": 1,
    "abs_tol": 1e-8,
    "rel_tol": 1e-9,
    "isunloading": False,
    "isunloadingonly": False,
    "ispctrl": True,
    "epiid_Kadj_coeff": [30, 10],
    "active_region": [0],
    "annulus_region": [1],
    "annulus_stiffness_factor": 100,
    "dashpotparam": [10.0e1,2.0e1],
    "permeability": 1.0e-9,
    "p_a": 0.0,
    "p_v": 1300.0,
    "beta_a": 3.5e-5,
    "beta_v": 3.0e-5,
    # Purkinje fiber network parameters:
    "isPJ": False, # False: LBBB / True: PJ activation
    "pj_tnodes": "PJ.csv",
    "PJ_current_intensity": 25.0,   
    "d_pj": 150,
    "ploc_mode": False,
    "pacing_timing": [[0.0, 10.0]],    
    "ploc": [[-0.574335, -1.8842, -0.168375]], # -> HIS
    # "ploc": [[-0.301386, -1.6874, -6.94474]], # -> Epi apex 
    # "ploc": [[-0.334257, -2.79273, -3.64153]],  # -> Epi mid    
    # "ploc": [[-0.305474, -3.09755, -0.874514]],  # -> Epi base  
    "ploc_tol": 0.002, #PMJ Purkinje endnodes to myocytes
    # Conduction Block:
    "lbbb": False,
    "lbbb_delay": 0.0, # No conduction after lbbb_location node
    "lbbb_location" : 8, #node location in pj network
    # Myocardium parameters:
    "isLV": True,
    "isBiV": False, 
    "current_intensity": 1.0,       
    "d_iso": 0.008,
    "d_ani_factor": 0.15,
    "ani_factor": 1000.0,
    "HIS" : [[-0.574335, -1.8842, -0.168375]], 
    "Ischemia": False,
    "GiccioneParams": GuccioneParams,
}

# Run Simulation
run_BiV_ClosedLoop(IODet=IODetails, SimDet=SimDetails)
#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

# Postprocessing Mechanics + EP intividual cases: 
# dumpvtk(IODet=IODetails, SimDet=SimDetails, ME_var=[["u", "CG", 1]], EP_var=[["phi", "CG", 1]], PJ_var=[["phi", "CG", 1]])
# plothemodynamics(IODet=IODetails, SimDet=SimDetails)
# plotpressure(IODet=IODetails, SimDet=SimDetails, compartment='lv')
# compute_activation(IODet=IODetails, SimDet=SimDetails)
# postprocessdata(IODet=IODetails, SimDet=SimDetails)
#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

# Postprocessing Haemodynamic comparison:
# plot_PV_comparison(IODet=IODetails, SimDet=SimDetails, cycle=None)
# plothemodynamics_comparison(IODet=IODetails, SimDet=SimDetails, cycle=None)
# plot_PV_comparison_ind(IODet=IODetails, SimDet=SimDetails, cycle=Circparam["stop_iter"])
# plothemodynamics_comparison_ind(IODet=IODetails, SimDet=SimDetails, cycle=Circparam["stop_iter"])
#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

# Postprocessing Strain global/hemisphere/AHA:
# compute_strain(IODet=IODetails, SimDet=SimDetails, LVid = 0)
# compute_strain_split(IODet=IODetails, SimDet=SimDetails, LVid = 0, axis_split=('x', 0.0))
# compute_strain_AHA_profile(IODet=IODetails, SimDet=SimDetails, LVid = 0, cycle=Circparam["stop_iter"])
# strain_plot_AHA(IODet=IODetails,SimDet=SimDetails,mode='all')
# plot_min_bullseye_per_cycle(IODet=IODetails, SimDet=SimDetails)
# check_AHA_plot(IODet=IODetails, SimDet=SimDetails, target_segment=1, R_apex=1.0)

