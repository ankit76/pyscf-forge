from pyscf import afqmc, cc, fci, gto, scf

nh = 8
r = 2
atomstring = ""
for i in range(nh):
    atomstring += f"H 0 0 {i * r}\n"
mol = gto.M(atom=atomstring, basis="sto-6g", unit="Bohr")

# fci for reference
mf = scf.RHF(mol)
mf.kernel()
ci = fci.FCI(mf)
e, _ = ci.kernel()
print(f"FCI energy: {e}")

# uccsd for trial
umf = scf.UHF(mol)
dm0 = 0.0 * umf.get_init_guess()
for i in range(nh // 2):
    dm0[0][2 * i, 2 * i] = 1.0
    dm0[1][2 * i + 1, 2 * i + 1] = 1.0
umf.kernel(dm0)
mycc = cc.UCCSD(umf)
mycc.kernel()
# uccsd(t) for reference
et = mycc.ccsd_t()
print(f"CCSD(T) energy: {mycc.e_tot + et}")

# afqmc
myafqmc = afqmc.AFQMC(mycc)
myafqmc.seed = 285055
myafqmc.n_blocks = 50
myafqmc.kernel()
