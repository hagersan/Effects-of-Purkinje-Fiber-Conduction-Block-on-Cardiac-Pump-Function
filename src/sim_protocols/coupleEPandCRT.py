import dolfin
from dolfin import *
import sys,pdb
sys.path.append("/mnt/Research")
from fenicstools import *
import numpy as np

class coupleEPandCRT(object):
    """
    Coupling between externally defined CRT pacing sites and the EP model.
    Activates myocardial pacing at specified CRT positions using fixed timing windows.
    """    

    def __init__(self, EPmodel, EPparams, state_obj):
        self.EPmodel = EPmodel
        self.EPparams = EPparams
        self.state_obj = state_obj

        # CRT pacing node coordinates
        self.crt_nodes = np.array(EPparams["CRT_pos"])    
        self.crt_intensity = EPparams["CRT_current_intensity"]
        self.crt_timing = np.array(EPparams["CRT_pacing_timing"])  

        # --- Skip if CRT inactive ---
        if len(self.crt_nodes) == 0 or self.crt_intensity == 0.0:
            info("CRT inactive or empty; pacing will remain off.")
            self.fstim_idx = []
            return
        
        # --- Determine CRT indices (after PJ sites) ---
        nPJ  = EPparams.get("nPJ", 0)
        nCRT = EPparams.get("nCRT", len(self.crt_nodes))
        self.fstim_idx = list(range(nPJ, nPJ + nCRT))
        info(f"✅ CRT pacing will control fstim indices {self.fstim_idx}")

        self.comms = EPmodel.mesh.mpi_comm()


    # -------------------------------------------------------------------------
    def UpdateCRTandEP(self):
        """
        Activates CRT pacing sources (fstim * delta) during timing windows.
        """
        if not self.fstim_idx:
            return

        EPmodel = self.EPmodel
        t = self.state_obj.t

        # Check if we are inside any active pacing window
        active_now = any(
            (t_on <= t <= t_on + dur) for (t_on, dur) in self.crt_timing if dur > 0.0
        )

        stim_val = self.crt_intensity if (active_now and self.crt_intensity > 0.0) else 0.0

        # Update CRT fstim entries
        for idx in self.fstim_idx:
            EPmodel.fstim_array[idx].iStim = stim_val

        # Optional hook for EP model
        if hasattr(EPmodel, "UpdateActivation"):
            EPmodel.UpdateActivation()

        # print(f"[CRT] t={t:.3f}, timing windows={self.crt_timing}")
        # active_now = any(
        #     (t_on <= t <= t_on + dur)
        #     for (t_on, dur) in self.crt_timing if dur > 0.0
        # )
        # print(f"[CRT] active_now={active_now}")
        # stim_val = self.crt_intensity if (active_now and self.crt_intensity > 0.0) else 0.0
        # print(f"[CRT] stim_val={stim_val}, fstim_idx={self.fstim_idx}")


    # -------------------------------------------------------------------------
    def reset(self):
        """
        Resets CRT pacing states between cardiac cycles.
        """
        if not self.fstim_idx:
            return

        for idx in self.fstim_idx:
            self.EPmodel.fstim_array[idx].iStim = 0.0
        return