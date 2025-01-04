# Reference code: https://github.com/Wells-Group/TEAM30

from typing import Dict, Tuple
import basix.ufl
import numpy as np
import ufl
from dolfinx import cpp, default_scalar_type, fem
from mpi4py import MPI
from petsc4py import PETSc

class MagneticField2D():
    def __init__(self, AzV: fem.Function,
                 form_compiler_options: dict = {}, jit_parameters: dict = {}):
        """
        Class for interpolate the magnetic vector potential (here as the first part of the mixed function AvZ)
        to the magnetic flux intensity B=curl(A)

        Parameters
        ==========
        AzV
            The mixed function of the magnetic vector potential Az and the Scalar electric potential V

        form_compiler_options
            Parameters used in FFCx compilation of this form. Run `ffcx --help` at
            the commandline to see all available options. Takes priority over all
            other parameter values, except for `scalar_type` which is determined by
            DOLFINx.

        jit_parameters
            Parameters used in CFFI JIT compilation of C code generated by FFCx.
            See `python/dolfinx/jit.py` for all available parameters.
            Takes priority over all other parameter values.
        """
        degree = AzV.function_space.ufl_element().degree()
        mesh = AzV.function_space.mesh
        cell = mesh.ufl_cell()

        # Create dolfinx Expression for electromagnetic field B (post processing)
        # Use minimum DG 1 as VTXFile only supports CG/DG>=1
        el_B = basix.ufl.element("DG", cell.cellname(),
                                 max(degree - 1, 1),
                                 shape=(mesh.geometry.dim,),
                                 gdim=mesh.geometry.dim)
        VB = fem.FunctionSpace(mesh, el_B)
        self.B = fem.Function(VB)
        B_2D = ufl.as_vector((AzV[0].dx(1), -AzV[0].dx(0)))
        self.Bexpr = fem.Expression(B_2D, VB.element.interpolation_points(),
                                    form_compiler_options=form_compiler_options,
                                    jit_options=jit_parameters)

    def interpolate(self):
        """
        Interpolate magnetic field
        """
        self.B.interpolate(self.Bexpr)


def update_current_density(J_0: fem.Function, omega: float, t: float, ct: cpp.mesh.MeshTags_int32,
                           currents: Dict[np.int32, Dict[str, float]]):
    """
    Given a DG-0 scalar field J_0, update it to be alpha*J*cos(omega*t + beta)
    in the domains with copper windings
    """
    J = 1413810.0970277672
    J_0.x.array[:] = 0
    for domain, values in currents.items():
        _cells = ct.find(domain)
        J_0.x.array[_cells] = np.full(len(_cells), J * values["alpha"]
                                      * np.cos(omega * t + values["beta"]))


def update_magnetization(Mvec, coercivity, omega_u, t, ct, domains, pm_orientation):
        block_size = 2 # Mvec.function_space.dofmap.index_map_bs = 2 for 2D
        coercivity = 8.38e5  # [A/m]   
        sign = 1

        for (material, domain) in domains.items():
            if material == 'PM':
                for marker in domain:
                    if marker in [13, 15, 17, 19, 21]:
                        inout = 1
                    elif marker in [14, 16, 18, 20, 22]:
                        inout = -1
                    angle = pm_orientation[marker] + omega_u * t
                    Mx = coercivity * np.cos(angle) * sign * inout
                    My = coercivity * np.sin(angle) * sign * inout

                    cells = ct.find(marker)
                    for cell in cells:
                        idx = block_size * cell
                        Mvec.x.array[idx + 0] = Mx
                        Mvec.x.array[idx + 1] = My

        # Mvec.x.scatter_forward()
        Mvec.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                mode=PETSc.ScatterMode.FORWARD)
