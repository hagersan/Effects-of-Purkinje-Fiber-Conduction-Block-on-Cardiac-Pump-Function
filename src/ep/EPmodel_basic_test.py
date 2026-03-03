# Electrophysiology model using Aliev Panilfov model
from dolfin import *
import numpy as np
from ..utils.nsolver import NSolver as NSolver

# from fenicstools import *
from ufl import indices
import dolfin as dolfin
from mpi4py import MPI as pyMPI
from ..utils.oops_objects_MRC2 import build_active_indicator
import pdb

class EPmodel(object):
    def __init__(self, params):
        self.parameters = self.default_parameters()
        self.parameters.update(params)
        self.mesh = self.parameters["EPmesh"]

        if "ploc" in list(self.parameters.keys()):
            self.max_pace_label = len(self.parameters["ploc"])
            self.ploc = self.parameters["ploc"]
        else:
            self.max_pace_label = 0
            self.ploc = None

        self.ischemia_mask = self.parameters.get("Ischemia_mask", None)

        # if self.ischemia_mask is not None:
        #     try:
        #         # # ----------------------------------------------------------------------
        #         # # Build active indicator (1.0 = healthy, 0.0 = ischemic)
        #         # # ----------------------------------------------------------------------
        #         # V0 = FunctionSpace(self.mesh, "DG", 0)
        #         # self.active_indicator = Function(V0)
        #         # self.active_indicator.vector()[:] = 1.0  # default healthy

        #         # dm = V0.dofmap()
        #         # owned_first, owned_last = dm.ownership_range()
        #         # local_vals = self.active_indicator.vector().get_local()

        #         # n_local_isch = 0

        #         # for cell in cells(self.mesh):
        #         #     dof = dm.cell_dofs(cell.index())[0]

        #         #     if owned_first <= dof < owned_last:
        #         #         isch = int(self.ischemia_mask[cell.index()])  # MUST use cell.index()

        #         #         if isch == 1:
        #         #             local_vals[dof - owned_first] = 0.0  # ischemic
        #         #             n_local_isch += 1
        #         #         else:
        #         #             local_vals[dof - owned_first] = 1.0  # healthy

        #         # self.active_indicator.vector().set_local(local_vals)
        #         # self.active_indicator.vector().apply("insert")

        #         # n_global_isch = MPI.sum(self.mesh.mpi_comm(), n_local_isch)

        #         # if self.mesh.mpi_comm().rank == 0:
        #         #     print(f"[EP] Loaded {n_global_isch} ischemic cells (active=0)")

        #         V0 = FunctionSpace(self.mesh, "DG", 0)
        #         # self.active_indicator = Function(V0)
        #         # self.active_indicator = build_active_indicator(self.mesh, self.ischemia_mask)

        #         mask_array = np.array(self.ischemia_mask.array(), dtype=float)
        #         # self.active_indicator.vector()[:] = 1.0 - mask_array
                
        #         # ----------------------------------------------------------------------
        #         # Build active indicator (1.0 = healthy, 0.0 = ischemic)
        #         # WITHOUT build_active_indicator() — simple and MPI-safe
        #         # ----------------------------------------------------------------------
        #         V0 = FunctionSpace(self.mesh, "DG", 0)
        #         self.active_indicator = Function(V0)

        #         dm = V0.dofmap()
        #         owned_range = dm.ownership_range()
        #         owned_first, owned_last = owned_range
        #         local_size = owned_last - owned_first

        #         # Allocate local DOF values
        #         local_vals = np.zeros(local_size, dtype=float)

        #         # Loop over cells — fill indicator
        #         for cell in cells(self.mesh):
        #             dof = dm.cell_dofs(cell.index())[0]
        #             if owned_first <= dof < owned_last:
        #                 val = float(self.ischemia_mask[cell.index()])
        #                 # healthy = 1.0, ischemic = 0.0
        #                 local_vals[dof - owned_first] = 1.0 - val

        #         # Assign to vector
        #         self.active_indicator.vector().set_local(local_vals)
        #         self.active_indicator.vector().apply('insert')

        #         # Diagnostics
        #         n_local_isch = sum(1 for c in cells(self.mesh) if self.ischemia_mask[c] == 1)
        #         n_global_isch = MPI.sum(self.mesh.mpi_comm(), n_local_isch)                

        #         if self.mesh.mpi_comm().rank == 0:
        #             print(f"[EP] Loaded {int(mask_array.sum())} ischemic cells (active=0)")
        #         # pdb.set_trace()

        #     except Exception as e:
        #         print(f"[EP] Failed to build active_indicator: {e}")
        #         self.active_indicator = Constant(1.0)
        # else:
        #     self.active_indicator = Constant(1.0)

        self.active_indicator = build_active_indicator(mesh=self.mesh,ischemia_mask=self.ischemia_mask,enable_ischemia=self.parameters.get("Ischemia", False),label="EP")
        # pdb.set_trace()

        P1_ep = FiniteElement("CG", self.mesh.ufl_cell(), 1, quad_scheme="default")
        P1_ep._quad_scheme = "default"
        P2_ep = FiniteElement("DG", self.mesh.ufl_cell(), 0, quad_scheme="default")
        P2_ep._quad_scheme = "default"

        self.W_ep = FunctionSpace(self.mesh, MixedElement([P1_ep, P2_ep]))
        self.w_ep = Function(self.W_ep)
        self.dw_ep = TrialFunction(self.W_ep)
        self.wtest_ep = TestFunction(self.W_ep)
        self.w_n_ep = Function(self.W_ep)

        self.F_FHN, self.J_FHN, self.fstim_array = self.Problem()

    def default_parameters(self):
        return {"ploc": [[0.0, 0.0, 0.0, 100.0, 1]], "pacing_timing": [[1, 20.0]],}

    def calculateDmat(self, f0, s0, n0, mesh, mId):
        d_iso = self.parameters["d_iso"]  # 0.01 #0.02
        d_ani = self.parameters["d_ani"]  # 0.08 #0.1 #0.2
        ani_factor = self.parameters["ani_factor"]
        ischemic_factor = self.parameters["Ischemic_ep_factor"]
        d_Ani = d_ani
        dParam = {}

        D_iso = Constant(((d_iso, "0.0", "0.0"), ("0.0", d_iso, "0.0"), ("0.0", "0.0", d_iso)))

        i, j = indices(2)
        Dij = (d_Ani)*f0[i]*f0[j] + (d_Ani*1/ani_factor)*s0[i]*s0[j] + (d_Ani*1/ani_factor)*n0[i]*n0[j]+ D_iso[i,j]
        D_tensor = as_tensor(Dij, (i, j)) 

        # return D_tensor
        healthy_D = D_tensor
        ischemic_D = ischemic_factor * D_tensor  # or whatever factor you choose

        # ischemic_D = ischemic_factor * (D_tensor - D_iso) + D_iso


        if self.parameters["Ischemia"]:
            D_effective = self.active_indicator * healthy_D + (1 - self.active_indicator) * ischemic_D
        else:
            D_effective = healthy_D

        return D_effective        

    # def MarkStimulus(self):
    #     mesh = self.mesh
    #     ploc = self.parameters["ploc"]
    #     EpiBCid_ep = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    #     EpiBCid_ep.set_all(0)

    #     for ploc_ in ploc:
    #         class Omega_0(SubDomain):
    #             def inside(self, x, on_boundary):
    #                 return (
    #                     (x[0] - ploc_[0]) ** 2
    #                     + (x[1] - ploc_[1]) ** 2
    #                     + (x[2] - ploc_[2]) ** 2
    #                 ) <= ploc_[3] ** 2

    #         subdomain_0 = Omega_0()
    #         subdomain_0.mark(EpiBCid_ep, int(ploc_[4]))

    #     # Find maximum number of pace label
    #     self.max_pace_label = int(max(np.array(ploc)[:, 4]))
    #     return EpiBCid_ep

    def Problem(self):
        state_obj = self.parameters["state_obj"]

        if "f0" in list(self.parameters.keys()):
            f0_ep = self.parameters["f0"]

        if "s0" in list(self.parameters.keys()):
            s0_ep = self.parameters["s0"]

        if "n0" in list(self.parameters.keys()):
            n0_ep = self.parameters["n0"]

        AHAid_ep = self.parameters["AHAid"]
        matid_ep = self.parameters["matid"]
        if "facetboundaries" in list(self.parameters.keys()):
            facetboundaries_ep = self.parameters["facetboundaries"]
        else:
            facetboundaries_ep = None

        mesh = self.mesh

        W_ep = self.W_ep
        w_n_ep = self.w_n_ep
        w_ep = self.w_ep
        dw_ep = self.dw_ep
        wtest_ep = self.wtest_ep
        #state_obj = self.state_obj

        phi0 = interpolate(Expression("0.0", degree=0), W_ep.sub(0).collapse())
        r0 = interpolate(Expression("0.0", degree=0), W_ep.sub(1).collapse())

        assign(w_n_ep, [phi0, r0])

        phi_n, r_n = split(w_n_ep)
        phi, r = split(w_ep)

        bcs_ep = []

        phi_test, r_test = split(wtest_ep)

        alpha = Constant(0.01)
        g = Constant(0.002)
        b = Constant(0.15)
        c = Constant(8)
        mu1 = Constant(0.2)
        mu2 = Constant(0.3)

        def eps_FHN(phi, r):
            return g + (mu1 * r) / (mu2 + phi)

        def f_phi(phi, r):
            return -c * phi * (phi - alpha) * (phi - 1) - r * phi

        def f_r(phi, r):
            return eps_FHN(phi, r) * (-r - c * phi * (phi - b - 1.0))

        fhn_timeNormalizer = 12.9
        k = state_obj.dt / fhn_timeNormalizer
        f_phi = f_phi(phi, r)
        f_r = f_r(phi, r)

        comm = mesh.mpi_comm()  # MPI Communicator
        if 'interval' in mesh.ufl_cell()._cellname:
            if "lbbb" in list(self.parameters.keys()):
                if self.parameters["lbbb"]:
                    # Create a piecewise scaling Function
                    Dmat_scaling = Function(FunctionSpace(mesh, "DG", 0))
                    Dmat_values = Dmat_scaling.vector()
                    Dmat_values[:] = np.where(matid_ep.array() == 1, self.parameters["lbbb_delay"], 1.0)

                    # Set the final `Dmat` variable
                    Dmat = Constant(self.parameters["d_iso"]) * Dmat_scaling
                else:
                    Dmat = Constant(self.parameters["d_iso"])
            else:
                Dmat = Constant(self.parameters["d_iso"])
        else:
            D_tensor = self.calculateDmat(f0_ep, s0_ep, n0_ep, mesh=mesh, mId=AHAid_ep)
            Dmat = D_tensor

        self.fstim_array = []
        self.delta_array = []
        hmin = mesh.hmin()
        hmax = mesh.hmax()
        ploc_fct = 0.02
        self.havg = ploc_fct*(hmin + hmax)

        # Factor to enlarge the kernel for CRT electrodes
        CRT_eps_factor = self.parameters["CRT_eps_factor"] #2.0   # e.g. 3× wider kernel for CRT
        if self.parameters["CRT"]:
            n_CRT_electrodes = len(self.parameters["CRT_pos"])   # you said: exactly two electrodes        
        else:
            n_CRT_electrodes = 0

        # Indices of the last two pacing sites (treated as CRT electrodes)
        first_crt_idx = max(0, self.max_pace_label - n_CRT_electrodes)        

        self.cpp_code0 = """
            
            #include <pybind11/pybind11.h>
            #include <pybind11/eigen.h>
            #include <dolfin/function/Expression.h>
            #include <iostream>
            
            class test_exp : public dolfin::Expression {
              public:
                
                Eigen::VectorXd x0;
                double eps;
                
                test_exp() : dolfin::Expression() { }
        
                void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const {
                    double norm_squared = 0.0;
                    for (int i = 0; i < x0.size(); ++i)
                    {
                        norm_squared += (x[i] - x0[i]) * (x[i] - x0[i]);
                    }
                    //values[0] = eps /  std::sqrt(norm_squared + eps * eps);
                    values[0] = eps / M_PI / (norm_squared + eps * eps);
                    //std::cout << values[0] << " " <<  x[0] << " " <<  x[1] << " " << x[2] << std::endl;
                    //std::cout << values[0] << " " <<  x[0] << " " <<  x[1] << " " << x[2] << std::endl;
                }
        
            };
            
            PYBIND11_MODULE(SIGNATURE, m) {
                pybind11::class_<test_exp, std::shared_ptr<test_exp>, dolfin::Expression>
                (m, "test_exp")
                .def(pybind11::init<>())
                .def_readwrite("x0", &test_exp::x0)
                .def_readwrite("eps", &test_exp::eps)
                ;
            }
        """
        # pdb.set_trace()
        for p in np.arange(0, self.max_pace_label):
            fstim = Expression("iStim", iStim=0.000, degree=1)
            self.fstim_array.append(Expression("iStim", iStim=0.000, degree=1))
            x0 = self.ploc[p]
            delta = dolfin.CompiledExpression(dolfin.compile_cpp_code(self.cpp_code0).test_exp(), degree=1)

            # use larger eps ONLY for the last two pacing sites (CRT electrodes)
            if p >= first_crt_idx:
                delta.eps = CRT_eps_factor * self.havg
            else:
                delta.eps = self.havg

            # delta.eps = self.havg      
            delta.x0 = np.array(x0, dtype=float)
            self.delta_array.append(delta)
            #self.delta_array.append(Delta(eps=havg, x0=np.array(x0), degree=1))
            #delta = Delta(eps=havg, x0=np.array(x0), degree=1)
            #self.F_FHN -=  fstim * delta * phi_test * dx_ep

        dx_ep = dolfin.dx(mesh)

        self.F_FHN = (
            ((phi - phi_n) / k) * phi_test * dx_ep
            + dot(Dmat * grad(phi), grad(phi_test)) * dx_ep
            - f_phi * phi_test * dx_ep
            + ((r - r_n) / k) * r_test * dx_ep
            - f_r * r_test * dx_ep
        )

#        self.F_FHN = (
#           ((phi - phi_n)) * phi_test * dx_ep
#           + k * dot(lt * grad(phi), grad(phi_test)) * dx_ep
#           - k * f_phi * phi_test * dx_ep
#           + ((r - r_n)) * r_test * dx_ep
#           - k * f_r * r_test * dx_ep
#        )


        for fstim, delta in zip(self.fstim_array, self.delta_array):
            self.F_FHN -=  fstim * delta * phi_test * dx_ep
            #self.F_FHN -=  k * fstim * delta * phi_test * dx_ep
        # for fstim in self.fstim_array:            
        #     self.F_FHN -=  fstim  * phi_test * dx_ep

        self.J_FHN = derivative(self.F_FHN, w_ep, dw_ep)
        return (self.F_FHN, self.J_FHN, self.fstim_array,)

    def Solver(self):
        solverparams_FHN = {
            "Jacobian": self.J_FHN,
            "F": self.F_FHN,
            "w": self.w_ep,
            "boundary_conditions": [],
            "Type": 0,
            "mesh": self.mesh,
            "mode": 1,
            # --- Add tolerance forwarding here ---
            "abs_tol": self.parameters.get("abs_tol"),
            "rel_tol": self.parameters.get("rel_tol"),              
        }

        solver_FHN = NSolver(solverparams_FHN)
        return solver_FHN

    def UpdateVar(self):
        # Update EP variable
        self.w_n_ep.assign(self.w_ep)

        # Update Stimulus variable
        #self.state_obj = self.parameters["state_obj"]
        pace_time_array = self.parameters["pacing_timing"]

    def Reset(self):
        self.w_ep.assign(self.w_n_ep)

    def getphivar(self):
        phi_, r_ = self.w_n_ep.split(deepcopy=True)
        phi_.rename("phi_", "phi_")
        return phi_

    def getrvar(self):
        phi_, r_ = self.w_n_ep.split(deepcopy=True)
        r_.rename("r_", "r_")
        return r_

    def interpolate_potential_ep2me_phi(self, V_me):
        LagrangeInterpolator.interpolate(V_me, self.getphivar())
        return V_me

    def clamp_variables(self, lower_bound=-0.05):
        """Clamp phi and r components of the mixed Function w_ep."""
        phi, r = self.w_ep.split(deepcopy=True)

        # Clamp phi
        phi_vals = phi.vector().get_local()
        phi_vals = np.maximum(phi_vals, lower_bound)
        phi.vector().set_local(phi_vals)
        phi.vector().apply("insert")

        # Clamp r
        r_vals = r.vector().get_local()
        r_vals = np.maximum(r_vals, lower_bound)
        r.vector().set_local(r_vals)
        r.vector().apply("insert")

        # Re-assign to mixed function
        assign(self.w_ep, [phi, r])


    def reset(self):
        phi0 = interpolate(Expression("0.0", degree=0), self.W_ep.sub(0).collapse())
        r0 = interpolate(Expression("0.0", degree=0), self.W_ep.sub(1).collapse())
        assign(self.w_n_ep, [phi0, r0])
        assign(self.w_ep, [phi0, r0])
        return

class Delta(UserExpression):
    def __init__(self, eps, x0, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.x0 = x0

    def eval(self, values, x):
        eps = self.eps
        x0 = self.x0
        values[0] = eps/pi/(np.linalg.norm(x-x0)**2 + eps**2)

    def value_shape(self):
        return ()



