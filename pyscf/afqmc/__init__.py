from typing import Optional, Union

from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD

from pyscf import scf
from pyscf.afqmc import afqmc


def AFQMC(mf_or_cc: Union[scf.uhf.UHF, scf.rhf.RHF, scf.rohf.ROHF, CCSD, UCCSD]):
    return afqmc.AFQMC(mf_or_cc)


AFQMC.__doc__ = afqmc.AFQMC.__doc__


def run_afqmc(
    options: Optional[dict] = None,
    mpi_prefix: Optional[str] = None,
    nproc: Optional[int] = None,
    tmpdir: Optional[str] = None,
):
    print("# Assuming input files have been generated.")
    return afqmc.run_afqmc(
        options=options, mpi_prefix=mpi_prefix, nproc=nproc, tmpdir=tmpdir
    )


run_afqmc.__doc__ = afqmc.run_afqmc.__doc__
