include("myfloat.jl")

function associated_legendre(l::Int, m::Int, x::T) where {T <: AbstractFloat}
    (m < 0 || m > l) && return T(0)
    pmm = T(1)
    if m > 0
        somx2 = sqrt((T(1) - x) * (T(1) + x))   # = sin θ
        fact  = T(1)
        for _ in 1:m
            pmm  *= -fact * somx2
            fact += T(2)
        end
    end
    l == m && return pmm

    pmmp1 = T(2m + 1) * x * pmm
    l == m + 1 && return pmmp1

    pll = T(0)
    for ll in (m + 2):l
        pll   = (T(2ll - 1) * x * pmmp1 - T(ll + m - 1) * pmm) / T(ll - m)
        pmm   = pmmp1
        pmmp1 = pll
    end
    return pll
end

function spherical_harmonic_r(l::Int, m::Int, theta::T, phi::T) where {T <: AbstractFloat}
    abs_m = abs(m)
    fact_ratio = T(1)
    for k in (l - abs_m + 1):(l + abs_m)
        fact_ratio *= T(k)
    end

    norm = sqrt(T(2l + 1) / (T(4) * T(π)) / fact_ratio)
    m != 0 && (norm *= sqrt(T(2)))

    plm = associated_legendre(l, abs_m, cos(theta))
    m > 0  && return norm * plm * cos(T(m) * phi)
    m < 0  && return norm * plm * sin(T(-m) * phi)
    return norm * plm   # m == 0
end

struct Point{T}
    r::T
    theta::T
    phi::T
end

function Point(x::T, y::T, z::T) where {T <: AbstractFloat}
    r = sqrt(x*x + y*y + z*z)
    r == T(0) && return Point{T}(T(0), T(0), T(0))   
    return Point{T}(r, acos(z / r), atan(y, x))
end

point_scaled(p::Point{T}, a::T) where {T} = Point{T}(a * p.r, p.theta, p.phi)

r2(p::Point{T}) where {T} = p.r * p.r

function y_lm(l::Int, m::Int, n::Point{T}) where {T <: AbstractFloat}
    if n.r == T(0)
        return (l == 0 && m == 0) ? T(1) / (T(2) * sqrt(T(π))) : T(0)
    end
    return n.r^l * spherical_harmonic_r(l, m, n.theta, n.phi)
end


const BIGBOUND = 10.0   

function small_kernel(l::Int, m::Int, n::Point{T}, t::T) where {T <: AbstractFloat}
    π_t = T(π)
    return y_lm(l, m, point_scaled(n, T(2) * π_t)) * exp(-π_t * π_t * r2(n) / t)
end

function big_kernel(l::Int, m::Int, n::Point{T}, t::T) where {T <: AbstractFloat}
    return y_lm(l, m, n) * exp(-t * r2(n))
end

function inner_summand(l::Int, m::Int, n::Point{T}, q2::T) where {T <: AbstractFloat}
    return y_lm(l, m, n) / (r2(n) - q2)
end

function lattice_sum(func, low::T, high::T, l::Int, m::Int, param::T) where {T <: AbstractFloat}
    result = T(0)
    ubound = Int(trunc(high * high))
    lbound = Int(ceil(low * low))
    ibound = Int(trunc(high))  

    for i in -ibound:ibound
        for j in -ibound:ibound
            for k in -ibound:ibound
                mag = i*i + j*j + k*k
                if lbound <= mag <= ubound
                    result += func(l, m, Point(T(i), T(j), T(k)), param)
                end
            end
        end
    end
    return result
end

function zero_to_one(l::Int, m::Int, q2::T, t::T) where {T <: AbstractFloat}
    π_t = T(π)

    constterm = (l == 0 && m == 0) ? (T(4) * π_t)^(-2) * t^(T(-3) / T(2)) : T(0)

    i_val = if     l % 4 == 0;                T(1)
            elseif l % 4 == 1 || l % 4 == 3;  T(0)
            else;                             T(-1)   # l % 4 == 2
            end

    sq = sqrt(q2)   

    term1 = (i_val / (T(2) * t)^l) * (T(4) * π_t * t)^(T(-3) / T(2)) *
            lattice_sum(small_kernel, T(0), T(BIGBOUND), l, m, t)

    term2 = (T(2) * π_t)^(-3) *
            lattice_sum(big_kernel,   T(0), sq,          l, m, t)

    return (term1 - term2) * exp(t * q2) - constterm
end

function one_to_inf(l::Int, m::Int, q2::T, t::T) where {T <: AbstractFloat}
    π_t = T(π)
    return (T(2) * π_t)^(-3) *
           lattice_sum(big_kernel, sqrt(q2), T(BIGBOUND), l, m, t) *
           exp(t * q2)
end


identity_transform(func, l, m, q2::T, t::T) where {T} =
    func(l, m, q2, t)


exponential_transform(func, l, m, q2::T, t::T) where {T} =
    func(l, m, q2, -log(t)) / t


invsqrt_transform(func, l, m, q2::T, t::T) where {T} =
    T(2) * t * func(l, m, q2, t * t)

@enum MidptRule normal_rule cov_exponential cov_invsqrt

function midpoint_step!(func, transform,
                        l::Int, m::Int, q2::T, a::T, b::T,
                        n::Int, s_state::Ref{T}) where {T <: AbstractFloat}
    if n == 1
        pt = T(0.5) * (a + b)
        s_state[] = (b - a) * transform(func, l, m, q2, pt)
        return s_state[]
    end

    
    it   = 3^(n - 2)
    npts = T(it)
    h1   = (b - a) / (T(3) * npts)
    h2   = h1 + h1
    t    = a + T(0.5) * h1
    acc  = T(0)
    for _ in 1:it
        acc += transform(func, l, m, q2, t);  t += h2
        acc += transform(func, l, m, q2, t);  t += h1
    end
    s_state[] = (s_state[] + (b - a) * acc / npts) / T(3)
    return s_state[]
end

function romberg_integrate(NUM_PTS::Int, func,
                           l::Int, m::Int, q2::T,
                           aa::T, bb::T, rule::MidptRule) where {T <: AbstractFloat}
    MAX_STEPS = 14
    FRAC_ACC  = T(10)^(-4)

    svec = Vector{T}(undef, MAX_STEPS + 2)   
    hvec = Vector{T}(undef, MAX_STEPS + 2)   
    hvec[1] = T(1)

    a, b = if rule == cov_exponential;  T(0), exp(-aa)
           elseif rule == cov_invsqrt;  T(0), sqrt(bb)
           else;                        aa, bb
           end

    transform = if rule == cov_exponential;  exponential_transform
                elseif rule == cov_invsqrt;  invsqrt_transform
                else;                        identity_transform
                end

    s_state = Ref(T(0))

    for j in 1:MAX_STEPS
        svec[j] = midpoint_step!(func, transform, l, m, q2, a, b, j, s_state)

        if j >= NUM_PTS
            z0 = j - NUM_PTS   

            table_idx = 1
            diff = abs(hvec[z0 + 1])
            c = Vector{T}(undef, NUM_PTS + 1)
            d = Vector{T}(undef, NUM_PTS + 1)
            for i in 1:NUM_PTS
                dift = abs(hvec[z0 + i])
                if dift < diff
                    table_idx = i
                    diff = dift
                end
                c[i] = svec[z0 + i]
                d[i] = svec[z0 + i]
            end

            ans = svec[z0 + table_idx]
            table_idx -= 1            

            error_val = T(0)
            for mm in 1:(NUM_PTS - 1)
                for i in 1:(NUM_PTS - mm)
                    ho = hvec[z0 + i]
                    hp = hvec[z0 + i + mm]
                    w  = c[i + 1] - d[i]
                    abs(ho - hp) <= eps(T) &&
                        @warn "romberg_integrate: roundoff error in Neville step"
                    den  = w / (ho - hp)
                    d[i] = hp * den
                    c[i] = ho * den
                end
                if 2 * table_idx < (NUM_PTS - mm)
                    error_val = c[table_idx + 1]
                else
                    error_val = d[table_idx]
                    table_idx -= 1
                end
                ans += error_val
            end

            abs(error_val) < FRAC_ACC * abs(ans) && return ans
        end

        svec[j + 1] = svec[j]          
        hvec[j + 1] = hvec[j] / T(9)  
    end

    @warn "romberg_integrate: failed to converge (too many steps)"
    return T(0)
end

function luscher_zeta_integralterm(l::Int, m::Int, q2::T) where {T <: AbstractFloat}
    int1 = romberg_integrate(5, zero_to_one, l, m, q2, T(0),  T(1),  cov_invsqrt)
    int2 = romberg_integrate(5, one_to_inf,  l, m, q2, T(1),  T(50), cov_exponential)
    π_t  = T(π)
    constterm = (l == 0 && m == 0) ? -π_t^(-2) / T(8) : T(0)
    return (T(2) * π_t)^3 * (constterm + int1 + int2)
end

function lzprint(q2::T, sum1::T, intl::T, ans::T) where {T}
    abs(ans) > T(50) && print('#')
end

function _luscher_zeta(l::Int, m::Int, q2::T, reduced::Bool) where {T <: AbstractFloat}
    q2 < T(0) && @warn "luscher_zeta: nonnegative argument required (got $q2)"

    sumbound = reduced ? (q2 < T(1) ? T(0) : sqrt(q2 - T(1))) : sqrt(q2)

    sum1 = lattice_sum(inner_summand, T(0), sumbound, l, m, q2)
    intl = luscher_zeta_integralterm(l, m, q2)
    ans  = sum1 + intl
    lzprint(q2, sum1, intl, ans)
    return ans
end

function luscher_zeta(l::Int, m::Int, q2::T) where {T <: AbstractFloat}
    return _luscher_zeta(l, m, q2, false)
end

function reduced_luscher_zeta(l::Int, m::Int, q2::T) where {T <: AbstractFloat}
    return _luscher_zeta(l, m, q2, true)
end