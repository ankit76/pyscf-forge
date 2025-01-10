from typing import Union

from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD

from pyscf import scf
from pyscf.afqmc import afqmc


def AFQMC(mf_or_cc: Union[scf.uhf.UHF, scf.rhf.RHF, scf.rohf.ROHF, CCSD, UCCSD]):
    return afqmc.AFQMC(mf_or_cc)
