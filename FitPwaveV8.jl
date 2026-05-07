include("zetaV2.jl")

using LinearAlgebra, Printf, Plots
using IMinuit
using Interpolations

const ħc    = 0.197327
const mπ30  = 0.3052
const mπ21  = 0.2076
const mπphy = 0.13498

const alat = 0.07746
const L32  = 32 * alat
const L48  = 48 * alat

const mD32_30 = 1.9665535
const mD48_30 = 1.96599941
const mD32_21 = 1.90386405
const mD48_21 = 1.90034485

const en32_30P_all = [4.04885436, 4.16733479, 4.28518312, 4.40113796]
const cov32_30P_all = [
    6.84030163e-6  3.46952643e-6  2.95027861e-6  4.45016065e-6
    3.46952643e-6  5.80483416e-6  3.11255948e-6  3.18535875e-6
    2.95027861e-6  3.11255948e-6  9.32069218e-6  3.30770960e-6
    4.45016065e-6  3.18535875e-6  3.30770960e-6  1.84461346e-5
]
const en48_30P_all = [3.98535654, 4.03828591, 4.09396336, 4.14669150]
const cov48_30P_all = [
    2.63942747e-6  2.02177698e-6  1.84800797e-6  2.07885040e-6
    2.02177698e-6  4.56657014e-6  2.51873728e-6  2.52560811e-6
    1.84800797e-6  2.51873728e-6  1.09682578e-5  1.72254793e-6
    2.07885040e-6  2.52560811e-6  1.72254793e-6  8.90562794e-6
]
const en32_21P_all = [3.92897997, 4.04913697, 4.17568093, 4.29405828]
const cov32_21P_all = [
    1.13327216e-5  1.03217001e-5  9.59510650e-6  1.15944333e-5
    1.03217001e-5  1.35215228e-5  1.01842870e-5  1.06556724e-5
    9.59510650e-6  1.01842870e-5  1.38193246e-5  9.79769660e-6
    1.15944333e-5  1.06556724e-5  9.79769660e-6  1.92102679e-5
]
const en48_21P_all = [3.85528288, 3.90877699, 3.96745067, 4.02146229]
const cov48_21P_all = [
    1.52376297e-6  9.89268333e-7  9.33232355e-7  1.18979689e-6
    9.89268333e-7  2.68513480e-6  1.18606528e-6  1.09438700e-6
    9.33232355e-7  1.18606528e-6  3.07052741e-6  8.38670993e-7
    1.18979689e-6  1.09438700e-6  8.38670993e-7  4.54340518e-6
]

const NP32 = 1
const NP48 = 3

const en32_30P = en32_30P_all[1:NP32]
const en48_30P = en48_30P_all[1:NP48]
const en32_21P = en32_21P_all[1:NP32]
const en48_21P = en48_21P_all[1:NP48]

const σ32_30P = sqrt.(diag(cov32_30P_all[1:NP32, 1:NP32]))
const σ48_30P = sqrt.(diag(cov48_30P_all[1:NP48, 1:NP48]))
const σ32_21P = sqrt.(diag(cov32_21P_all[1:NP32, 1:NP32]))
const σ48_21P = sqrt.(diag(cov48_21P_all[1:NP48, 1:NP48]))

const a1_ref_30 = 12.0   # fm³
const r1_ref_30 = 20.1   # fm⁻¹
const a1_ref_21 =  6.0   # fm³
const r1_ref_21 = 11.9   # fm⁻¹

if !@isdefined(zeta_interp)
    println("预计算 Lüscher zeta 插值表...")
    global q_grid      = range(0.001, 3.0, length=5000)
    global Z_grid      = luscher_zeta.(0, 0, collect(q_grid))
    global zeta_interp = linear_interpolation(q_grid, Z_grid)
end


function cotdelta_lus_P(E::Float64, mD::Float64, L::Float64)
    p2 = (E/2)^2 - mD^2
    p2 ≤ 0 && return NaN
    p_fm = sqrt(p2) / ħc
    q    = p_fm * L / (2π)
    return zeta_interp(q^2) / (π^1.5 * q)
end

function p3cotdelta_lus(E::Float64, mD::Float64, L::Float64)
    p2 = (E/2)^2 - mD^2
    p2 ≤ 0 && return NaN
    p_fm = sqrt(p2) / ħc
    return p_fm^3 * cotdelta_lus_P(E, mD, L)
end

function p3cotdelta_ere(E::Float64, mD::Float64, a1::Float64, r1::Float64)
    p2 = (E/2)^2 - mD^2
    p2 ≤ 0 && return NaN
    p = sqrt(p2) / ħc
    return 1.0/a1 + 0.5*r1*p^2
end

function dp3cotdelta_dE(E::Float64, mD::Float64, L::Float64)
    dE = 1e-6
    cp = p3cotdelta_lus(E + dE, mD, L)
    cm = p3cotdelta_lus(E - dE, mD, L)
    (isnan(cp) || isnan(cm)) && return NaN
    return (cp - cm) / (2dE)
end

function chi2_P(x::Vector,
                en32, cov32, mD32, n32, L32_,
                en48, cov48, mD48, n48, L48_)::Float64
    a1, r1 = x[1], x[2]
    abs(a1) < 1e-6 && return 1e12

    J32 = [dp3cotdelta_dE(en32[k], mD32, L32_) for k in 1:n32]
    J48 = [dp3cotdelta_dE(en48[k], mD48, L48_) for k in 1:n48]
    (any(isnan, J32) || any(isnan, J48)) && return 1e12

    C32 = Diagonal(J32) * cov32 * Diagonal(J32)
    C48 = Diagonal(J48) * cov48 * Diagonal(J48)

    r32 = [p3cotdelta_lus(en32[k], mD32, L32_) - p3cotdelta_ere(en32[k], mD32, a1, r1)
           for k in 1:n32]
    r48 = [p3cotdelta_lus(en48[k], mD48, L48_) - p3cotdelta_ere(en48[k], mD48, a1, r1)
           for k in 1:n48]
    (any(isnan, r32) || any(isnan, r48)) && return 1e12

    return dot(r32, inv(C32) * r32) + dot(r48, inv(C48) * r48)
end


function chi2_r1only(x::Vector, a1_fixed::Float64,
                     en32, cov32, mD32, n32, L32_,
                     en48, cov48, mD48, n48, L48_)::Float64
    r1 = x[1]
    return chi2_P([a1_fixed, r1], en32, cov32, mD32, n32, L32_,
                                  en48, cov48, mD48, n48, L48_)
end

function chiral_extrap(v30::Float64, v21::Float64)
    Δ  = mπ30^2 - mπ21^2
    c1 = (v30 - v21) / Δ
    c0 = v21 - c1 * mπ21^2
    return c0, c1, c0 + c1 * mπphy^2
end

println("\n" * "="^65)
println("A：同时拟合 a₁ 和 r₁")
println("="^65)


for (label, mπ_val, en32, cov32, mD32, en48, cov48, mD48,
     a1_ref, r1_ref) in [
    ("mπ≈305 MeV", mπ30,
     en32_30P, cov32_30P_all[1:NP32,1:NP32], mD32_30,
     en48_30P, cov48_30P_all[1:NP48,1:NP48], mD48_30,
     a1_ref_30, r1_ref_30),
    ("mπ≈207 MeV", mπ21,
     en32_21P, cov32_21P_all[1:NP32,1:NP32], mD32_21,
     en48_21P, cov48_21P_all[1:NP48,1:NP48], mD48_21,
     a1_ref_21, r1_ref_21),
]
    println("\n── $label ──")
    res = Minuit(
        x -> chi2_P(x, en32, cov32, mD32, NP32, L32,
                       en48, cov48, mD48, NP48, L48),
        [a1_ref, r1_ref];
        name     = ["a1", "r1"],
        error    = [abs(a1_ref)+1.0, 5.0],
        limit_a1 = (-15.0, 15.0),
        limit_r1 = (-100.0, 100.0),
    )
    res.errordef = 1.0
    res.strategy = 2
    migrad(res)
    minos(res)

    a1_fit, r1_fit = res.values
    χ2 = res.fval
    ndof = NP32 + NP48 - 2
    @printf("  a₁ = %+.4f fm³  (论文: %.1f)\n", a1_fit, a1_ref)
    @printf("  r₁ = %+.4f fm⁻¹ (论文: %.1f)\n", r1_fit, r1_ref)
    @printf("  χ²/dof = %.4f / %d = %.4f\n", χ2, ndof, χ2/ndof)
    me = res.merrors
a1_fit = res.values[1]
at_upper = abs(a1_fit - 15.0) < 0.01
at_lower = abs(a1_fit - (-15.0)) < 0.01
    if label == "mπ≈305 MeV"
        global a1_fit_30, r1_fit_30 = a1_fit, r1_fit
    else
        global a1_fit_21, r1_fit_21 = a1_fit, r1_fit
    end
end

println("\n" * "="^65)
println("B：固定 a₁（论文中心值），拟合 r₁")
println("="^65)

r1_30_B = r1_21_B = 0.0

for (label, a1_fix, r1_init,
     en32, cov32, mD32,
     en48, cov48, mD48,
     r1_ref) in [
    ("mπ≈305 MeV", a1_ref_30, r1_ref_30,
     en32_30P, cov32_30P_all[1:NP32,1:NP32], mD32_30,
     en48_30P, cov48_30P_all[1:NP48,1:NP48], mD48_30,
     r1_ref_30),
    ("mπ≈207 MeV", a1_ref_21, r1_ref_21,
     en32_21P, cov32_21P_all[1:NP32,1:NP32], mD32_21,
     en48_21P, cov48_21P_all[1:NP48,1:NP48], mD48_21,
     r1_ref_21),
]
    println("\n── $label（固定 a₁ = $a1_fix fm³）──")
    res = Minuit(
        x -> chi2_r1only(x, a1_fix,
                         en32, cov32, mD32, NP32, L32,
                         en48, cov48, mD48, NP48, L48),
        [r1_init];
        name     = ["r1"],
        error    = [5.0],
        limit_r1 = (-100.0, 100.0),
    )
    res.errordef = 1.0
    res.strategy = 2
    migrad(res)
    minos(res)

    r1_fit = res.values[1]
    χ2     = res.fval
    ndof   = NP32 + NP48 - 1
    @printf("  r₁ = %+.4f fm⁻¹ (论文: %.1f)\n", r1_fit, r1_ref)
    @printf("  χ²/dof = %.4f / %d = %.4f\n", χ2, ndof, χ2/ndof)
    me = res.merrors

    if label == "mπ≈305 MeV"
        global r1_30_B = r1_fit
    else
        global r1_21_B = r1_fit
    end
end

c0r1, c1r1, r1_phy = chiral_extrap(r1_30_B, r1_21_B)

println()
println("="^65)
println("方案B 手征外推（固定 a₁，外推 r₁）")
println("="^65)
@printf("  c₀ʳ¹ = %+.4f fm⁻¹,  c₁ʳ¹ = %+.4f fm⁻¹/GeV²\n", c0r1, c1r1)
@printf("  mπ≈305 MeV:    r₁ = %+.4f fm⁻¹\n", r1_30_B)
@printf("  mπ≈207 MeV:    r₁ = %+.4f fm⁻¹\n", r1_21_B)
@printf("  物理点(135MeV): r₁ = %+.4f fm⁻¹\n", r1_phy)
@printf("  手征极限(mπ→0): r₁ = %+.4f fm⁻¹\n", c0r1)

# ================================================================
# 相移散点
# ================================================================
function extract_delta(E::Float64, mD::Float64, L::Float64)::Float64
    p2 = (E/2)^2 - mD^2
    p2 ≤ 0 && return NaN
    p_fm = sqrt(p2) / ħc
    q    = p_fm * L / (2π)
    cotδ = luscher_zeta(0, 0, q^2) / (π^1.5 * q)
    return atand(1.0 / cotδ)
end

function delta_error(E::Float64, σE::Float64, mD::Float64, L::Float64)::Float64
    dE = max(σE * 1e-3, 1e-8)
    δp = extract_delta(E + dE, mD, L)
    δm = extract_delta(E - dE, mD, L)
    (isnan(δp) || isnan(δm)) && return NaN
    return abs((δp - δm) / (2dE)) * σE
end

δ32_30P  = extract_delta.(en32_30P, mD32_30, L32)
δ48_30P  = extract_delta.(en48_30P, mD48_30, L48)
δ32_21P  = extract_delta.(en32_21P, mD32_21, L32)
δ48_21P  = extract_delta.(en48_21P, mD48_21, L48)
σδ32_30P = delta_error.(en32_30P, σ32_30P, mD32_30, L32)
σδ48_30P = delta_error.(en48_30P, σ48_30P, mD48_30, L48)
σδ32_21P = delta_error.(en32_21P, σ32_21P, mD32_21, L32)
σδ48_21P = delta_error.(en48_21P, σ48_21P, mD48_21, L48)

println("\n散点相移值：")
println("  F32P30: δ₁ = ", round.(δ32_30P, digits=2), " °")
println("  F48P30: δ₁ = ", round.(δ48_30P, digits=2), " °")
println("  F32P21: δ₁ = ", round.(δ32_21P, digits=2), " °")
println("  F48P21: δ₁ = ", round.(δ48_21P, digits=2), " °")

# ================================================================
# ERE 曲线
# ================================================================
function δ_ere_P(E::Float64, mD::Float64, a1::Float64, r1::Float64)::Float64
    p2 = (E/2)^2 - mD^2
    p2 ≤ 0 && return NaN
    p      = sqrt(p2) / ħc
    p3cotδ = 1.0/a1 + 0.5*r1*p^2
    cotδ   = p3cotδ / p^3
    return atand(1.0 / cotδ)
end

mD_30 = (mD32_30 + mD48_30) / 2
mD_21 = (mD32_21 + mD48_21) / 2

Ec30 = collect(range(2mD_30 + 0.003, 2mD_30 + 0.200, length=600))
Ec21 = collect(range(2mD_21 + 0.003, 2mD_21 + 0.200, length=600))

# 方案A 曲线（固定 a1 = 论文值，拟合 r1）
δP_B30 = δ_ere_P.(Ec30, mD_30, a1_ref_30, r1_30_B)
δP_B21 = δ_ere_P.(Ec21, mD_21, a1_ref_21, r1_21_B)
# 论文参考曲线
δP_ref_30 = δ_ere_P.(Ec30, mD_30, a1_ref_30, r1_ref_30)
δP_ref_21 = δ_ere_P.(Ec21, mD_21, a1_ref_21, r1_ref_21)

# ================================================================
# 作图
# ================================================================
gr()

function make_panel(E_curve, δ_fit, δ_ref,
                    E32, δ32, σδ32,
                    E48, δ48, σδ48,
                    a1_val, r1_fit, r1_ref,
                    mπ_label, xlim_lo, xlim_hi)

    plt = plot(E_curve, δ_fit;
        color=:red, linewidth=2.5, linestyle=:solid,
        label=@sprintf("a₁=%.1f, r₁=%+.2f", a1_val, r1_fit),
        xlabel="E  [GeV]", ylabel="δ₁  [°]",
        title="P-wave  I(Jᴾ)=0(1⁻)  mπ≈$(mπ_label)  (Z=1)",
        xlims=(xlim_lo, xlim_hi), ylims=(-10, 45),
        legend=:topleft, framestyle=:box,
        grid=true, gridstyle=:dash, gridcolor=:lightgray,
        tickfontsize=10, guidefontsize=11,
        legendfontsize=8, titlefontsize=10,
        size=(680, 500), dpi=150, margin=5Plots.mm)

    plot!(plt, E_curve, δ_ref;
        color=:darkorange, linewidth=2.0, linestyle=:dash,
        label=@sprintf("a₁=%.1f, r₁=%.1f", a1_val, r1_ref))

    hline!(plt, [0]; color=:black, linestyle=:dash,
           alpha=0.3, linewidth=1, label="")

    scatter!(plt, E32, δ32;
        yerror=σδ32, marker=:square, markercolor=:blue,
        markersize=7, markerstrokewidth=1.5,
        linecolor=:blue, linewidth=1.5, label="T₁⁻  L=32")

    scatter!(plt, E48, δ48;
        yerror=σδ48, marker=:circle, markercolor=:green,
        markersize=7, markerstrokewidth=1.5,
        markerstrokecolor=:darkgreen,
        linecolor=:darkgreen, linewidth=1.5, label="T₁⁻  L=48")

    return plt
end

pP30 = make_panel(Ec30, δP_B30, δP_ref_30,
                  en32_30P, δ32_30P, σδ32_30P,
                  en48_30P, δ48_30P, σδ48_30P,
                  a1_ref_30, r1_30_B, r1_ref_30,
                  "305 MeV", 3.90, 4.15)

pP21 = make_panel(Ec21, δP_B21, δP_ref_21,
                  en32_21P, δ32_21P, σδ32_21P,
                  en48_21P, δ48_21P, σδ48_21P,
                  a1_ref_21, r1_21_B, r1_ref_21,
                  "207 MeV", 3.78, 4.05)

plt_all = plot(pP30, pP21; layout=(1,2), size=(1360, 500), dpi=150)
savefig(plt_all, "fig_Pwave_dual1_mpi.pdf")
display(plt_all)