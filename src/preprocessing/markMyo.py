import sys, os, pdb
import vtk
sys.path.append("/mnt/Research")
from dolfin import *
import numpy as np
import os

def markMyo_cells(IODet, SimDet):
    """
    Generates a NEW marked HDF5 file that is a full copy of the original
    myocardium HDF5 mesh but with an additional dataset:

        <casename_me_marked>/ischemia_mask

    The original file is untouched.

    Applies to both _me and _ep meshes.
    """

    ischemia_matid = set(SimDet.get("Ischemia_matid", [0, 1]))
    regions = SimDet.get("Ischemia_regions", [])
    if not regions:
        print("⚠️  No ischemic regions defined — skipping marking.")
        return

    # ------------------------------------------------------------
    # Process BOTH mechanics mesh (_me) and electrophysiology mesh (_ep)
    # ------------------------------------------------------------
    for mode in ["_me", "_ep"]:

        casename = IODet.get(f"casename{mode}")
        indir    = IODet.get(f"directory{mode}")
        casename_marked = IODet.get(f"casename{mode}_marked", casename + "_marked")

        if not casename or not indir:
            print(f"⚠️  Skipping {mode}: missing casename or directory.")
            continue

        infile  = os.path.join(indir, casename + ".hdf5")
        outfile = os.path.join(indir, casename_marked + ".hdf5")
        visfile = os.path.join(indir, f"myocardium_markedcells{mode}.pvd")

        print("\n=====================================================")
        print(f"  ISCHEMIA MARKING for {mode}")
        print("-----------------------------------------------------")
        print(f" Original file : {infile}")
        print(f" Marked file   : {outfile}")
        print(f" HDF5 group    : {casename_marked}")
        print("=====================================================\n")

        # ------------------------------------------------------------
        # Step 1 — Load original mesh
        # ------------------------------------------------------------
        mesh = Mesh()
        with HDF5File(mesh.mpi_comm(), infile, "r") as f:
            f.read(mesh, casename, False)

        dim = mesh.topology().dim()
        coords = mesh.coordinates()
        print(f"Mesh cells: {mesh.num_cells()}")

        # ------------------------------------------------------------
        # Step 2 — Load original matid
        # ------------------------------------------------------------
        matid = MeshFunction("size_t", mesh, dim, mesh.domains())
        with HDF5File(mesh.mpi_comm(), infile, "r") as f:
            if f.has_dataset(casename + "/matid"):
                f.read(matid, casename + "/matid")
                print(f"✔ Loaded matid from {casename}/matid")
            else:
                matid.set_all(0)
                print("⚠ No matid found; default=0")

        uniq = np.unique(matid.array())
        print(f" matid values present: {uniq}")

        # ------------------------------------------------------------
        # Step 3 — Build ischemia_mask
        # ------------------------------------------------------------
        cell_markers = MeshFunction("size_t", mesh, dim, 0)
        n_marked = 0

        print("\nMarking cells...")
        per_matid_count = {m: 0 for m in ischemia_matid}

        for cell in cells(mesh):
            mid = matid[cell]
            if mid not in ischemia_matid:
                continue

            vtx_ids = cell.entities(0)
            vtx_coords = coords[vtx_ids]

            inside = any(
                (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= r**2
                for (x, y, z) in vtx_coords
                for (cx, cy, cz, r) in regions
            )

            if inside:
                cell_markers[cell] = 1
                n_marked += 1
                per_matid_count[mid] += 1

        print(f"✔ Marked {n_marked} ischemic cells")
        print(f" Breakdown by matid: {per_matid_count}\n")

        # ------------------------------------------------------------
        # Step 4 — Create a NEW marked file by copying original file
        # ------------------------------------------------------------
        print("Copying all datasets from original file into marked file...")

        with HDF5File(mesh.mpi_comm(), infile, "r") as fin, \
             HDF5File(mesh.mpi_comm(), outfile, "w") as fout:

            # 1) Copy mesh
            fout.write(mesh, casename_marked)

            # 2) Copy every dataset from the original group
            orig_group = casename
            new_group  = casename_marked

            # List all datasets in the original group
            dataset_list = fin.attributes(orig_group).keys() if fin.attributes(orig_group) else []

            # We cannot use h5py, so we must attempt known datasets:
            known_sets = [
                "coordinates", "topology", "matid",
                "eF", "eS", "eN",
                "facetboundaries", "AHAid", "baseid"
            ]

            for ds in known_sets:
                path_in  = f"{orig_group}/{ds}"
                path_out = f"{new_group}/{ds}"
                if fin.has_dataset(path_in):
                    obj = None
                    try:
                        # Scalar meshfunctions need proper handling
                        if ds in ["matid", "facetboundaries", "AHAid", "baseid"]:
                            mf = MeshFunction("size_t", mesh, dim if ds=="matid" else 2, mesh.domains())
                            fin.read(mf, path_in)
                            fout.write(mf, path_out)
                        else:
                            fin.read(fout, path_in, path_out)
                    except:
                        pass

            # Add ischemia_mask
            fout.write(cell_markers, f"{new_group}/ischemia_mask")

        print(f"✔ New marked file written: {outfile}")

        # ------------------------------------------------------------
        # Step 5 — Save visualization
        # ------------------------------------------------------------
        File(visfile) << cell_markers
        print(f"✔ Saved visualization: {visfile}")
        print(f"=== DONE for {mode} ===\n")

    print("🎯 Finished generating NEW marked files for both meshes.\n")

# def markMyo_cells(IODet, SimDet):
#     """
#     Marks ischemic regions in both myocardium meshes (_me and _ep)
#     using pure FEniCS I/O (no h5py).

#     A cell is marked only if:
#        - Its matid[cell] ∈ SimDet['Ischemia_matid'] (e.g. LV or septum)
#        - At least one vertex lies inside a defined ischemic region
#     """

#     ischemia_matid = set(SimDet.get("Ischemia_matid", [0, 1]))  # Default LV + Septum
#     regions = SimDet.get("Ischemia_regions", [])
#     if not regions:
#         print("⚠️  No ischemic regions defined — skipping.")
#         return

#     for mode in ["_me", "_ep"]:
#         casename = IODet.get(f"casename{mode}")
#         indir    = IODet.get(f"directory{mode}")
#         if not casename or not indir:
#             print(f"⚠️  Skipping {mode}: missing casename or directory.")
#             continue

#         infile  = os.path.join(indir, casename + ".hdf5")
#         outfile = os.path.join(indir, casename + "_marked.hdf5")
#         visfile = os.path.join(indir, f"myocardium_markedcells{mode}.pvd")

#         print(f"\n=== Ischemia marking for {casename} ({mode}) ===")
#         print(f"Input : {infile}")

#         # --- Step 1: Load mesh ---
#         mesh = Mesh()
#         with HDF5File(mesh.mpi_comm(), infile, "r") as f:
#             f.read(mesh, casename, False)
#         dim = mesh.topology().dim()
#         print(f"Mesh cells: {mesh.num_cells()}")

#         # --- Step 2: Load matid directly ---
#         matid = MeshFunction("size_t", mesh, dim, mesh.domains())
#         with HDF5File(mesh.mpi_comm(), infile, "r") as f:
#             if f.has_dataset(casename + "/matid"):
#                 f.read(matid, casename + "/matid")
#                 print(f"✅  Loaded matid from {casename}/matid")
#             else:
#                 matid.set_all(0)
#                 print("⚠️  No matid found; defaulted to 0")

#         uniq = np.unique(matid.array())
#         print(f"   Unique matid values: {uniq} (0=LV, 1=Septum, 2=RV, 3=Annulus)")

#         # --- Step 3: Create ischemia markers ---
#         cell_markers = MeshFunction("size_t", mesh, dim, 0)
#         coords = mesh.coordinates()
#         n_marked = 0
#         per_id_count = {m: 0 for m in ischemia_matid}

#         for cell in cells(mesh):
#             mid = matid[cell]
#             if mid not in ischemia_matid:
#                 continue

#             vtx_ids = cell.entities(0)
#             vtx_coords = coords[vtx_ids]
#             inside = any(
#                 (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= r**2
#                 for (x, y, z) in vtx_coords
#                 for (cx, cy, cz, r) in regions
#             )

#             if inside:
#                 cell_markers[cell] = 1
#                 n_marked += 1
#                 per_id_count[mid] += 1

#         print(f"✅  Marked {n_marked} ischemic cells (matid ∈ {sorted(ischemia_matid)})")
#         print("   Breakdown by matid:", per_id_count)

#         # --- Step 4: Save to new HDF5 ---
#         with HDF5File(mesh.mpi_comm(), outfile, "w") as f:
#             f.write(mesh, casename + "_marked")
#             f.write(cell_markers, casename + "_marked/ischemia_mask")
#         print(f"💾  Saved ischemia mask to {outfile}")

#         # --- Step 5: Save visualization ---
#         File(visfile) << cell_markers
#         print(f"📊  Visualization file written to {visfile}")
#         print(f"=== Done for {mode} ===\n")

#     print("🎯  Completed ischemia marking for both _me and _ep meshes.\n")
