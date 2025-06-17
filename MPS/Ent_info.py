import torch as tc
import numpy as np
import matplotlib.pyplot as plt


# The number of random measurements to perform to find the minimum entropy.
mesure_time = 600
dtype = tc.complex128

print("Generate a random 2-qubit state...")
# We represent a 2-qubit state |ψ⟩ = Σᵢⱼ Cᵢⱼ |i⟩|j⟩, where i,j ∈ {0,1},
# as a 2x2 matrix `C` of its coefficients.
# `psi` is this coefficient matrix `C`.
psi = tc.randn((2, 2), dtype=dtype)
# We normalize the state such that its total probability is 1.
# This ensures ⟨ψ|ψ⟩ = Tr(C†C) = 1.
psi /= psi.norm()

print("Calculate entanglement entropy...")
# To describe the first qubit, we compute its reduced density matrix `ρ₀`.
# The formula is `ρ₀ = C C†`, where C† is the conjugate transpose of C.
# The code computes `rho = C* Cᵀ`, which is `(ρ₀)ᵀ`.
# This is a valid optimization because a matrix and its transpose have the same eigenvalues.
rho = psi.conj().mm(psi.t())

# The eigenvalues (`lm`, for lambda) of `ρ₀` are used to calculate entanglement entropy.
# These eigenvalues are also the squares of the Schmidt coefficients of the state.
lm = tc.linalg.eigvalsh(rho)

# The entanglement entropy is the von Neumann entropy of the reduced state.
# Formula: S = -Tr(ρ₀ log(ρ₀)) = -Σᵢ λᵢ log(λᵢ), where λᵢ are the eigenvalues.
# We filter out eigenvalues that are zero or very close to it to avoid log(0) = -inf.
ent = -lm.inner(tc.log(lm))

print("Implement random measurement on qubit-0...")
# We will store the entropy of each random measurement's outcome distribution here.
s = tc.zeros(mesure_time, dtype=lm.dtype)

# We perform many random measurements. The minimum entropy found from these
# measurements should approximate the true entanglement entropy `ent`.
for t in range(mesure_time):
    # --- Create a random measurement basis ---
    # To perform a measurement, we need an orthonormal basis for the qubit's state space.
    # We create one by generating a random Hermitian matrix `H = h + h†`...
    h = tc.randn((2, 2), dtype=lm.dtype)
    # ...and then creating a unitary matrix `U = exp(iH)`.
    # The columns of this unitary matrix `U` form a random orthonormal basis {|φ₀⟩, |φ₁⟩}.
    proj = tc.matrix_exp(1j * (h + h.t()))

    # --- Calculate the probability distribution ---
    # `p` will store the probabilities [p₀, p₁] of measuring the qubit in states
    # |φ₀⟩ and |φ₁⟩ respectively.
    # The vector `proj[:, s]` corresponds to the measurement basis vector `|φₛ⟩`.
    # The line below calculates pₛ = ⟨φₛ|ρ₀|φₛ⟩. This is the probability of the
    # state `ρ₀` collapsing to the measurement state `|φₛ⟩`.
    # This is mathematically equivalent to, but computationally more direct than,
    # first building the measurement operator Mₛ = |φₛ⟩⟨φₛ| and then computing Tr(ρ₀ Mₛ).
    p = [proj[:, s].conj().inner(rho).inner(proj[:, s]) for s in [0, 1]]

    # --- Calculate the entropy of the probability distribution ---
    # For each measurement, we get a classical probability distribution {p₀, p₁}.
    # We calculate the Shannon entropy of this distribution.
    # Formula: S_VN = -Σₛ pₛ ln(pₛ)
    s[t] = -(p[0].real * np.log(p[0].real) + p[1].real * np.log(p[1].real))


print(
    "On qubit-0:\n Maximum of entropy = %g \n "
    "Minimum of entropy = %g \n "
    "Entanglement entropy = %g" % (max(s), min(s), ent)
)

print("Implement random measurement on qubit-1...")
rho = psi.t().mm(psi.conj())
s1 = tc.zeros(mesure_time, dtype=lm.dtype)
for t in range(mesure_time):
    # 随机生成测量基
    h = tc.randn((2, 2), dtype=lm.dtype)
    proj = tc.matrix_exp(1j * (h + h.t()))
    # 计算概率分布
    p = [proj[:, s].conj().inner(rho).inner(proj[:, s]) for s in [0, 1]]
    # 计算冯诺伊曼熵
    s1[t] = -(p[0].real * np.log(p[0].real) + p[1].real * np.log(p[1].real))

print(
    "On qubit-1:\n Maximum of entropy = %g \n "
    "Minimum of entropy = %g \n "
    "Entanglement entropy = %g" % (max(s1), min(s1), ent)
)

x = np.arange(mesure_time)
fig = plt.figure()
fig.add_subplot(1, 2, 1)
ent_m = plt.scatter(x, s)
(ent_f,) = plt.plot(x, np.ones((mesure_time,)) * ent.item(), color="r", linestyle="--")
plt.xlabel("n-th random measurement on qubit-0")
plt.ylabel("entropy")
plt.legend([ent_m, ent_f], ["entropy by measurement", "entanglement entropy"])

fig.add_subplot(1, 2, 2)
ent_m = plt.scatter(x, s1)
(ent_f,) = plt.plot(x, np.ones((mesure_time,)) * ent.item(), color="r", linestyle="--")
plt.xlabel("n-th random measurement on qubit-1")
plt.legend([ent_m, ent_f], ["entropy by measurement", "entanglement entropy"])
plt.show()
