using SpecialFunctions  

const PRECISION_DIGITS = get(ENV, "PRECISION", "20") |> x -> parse(Int, x)
const PRECISION_BITS   = ceil(Int, PRECISION_DIGITS * log2(10))

setprecision(BigFloat, PRECISION_BITS)

const Myfloat = BigFloat   

function spherical_harmonic_r(l::Int, m::Int, theta::T, phi::T) where {T <: AbstractFloat}
    absm = abs(m)
    x    = cos(theta)
    plm  = legendre_p(l, absm, x)          
    norm = sqrt((2l + 1) / (4 * T(π)) *
                factorial(big(l - absm)) / factorial(big(l + absm)))
    if m >= 0
        return T(norm * plm * cos(m * phi))
    else
        return T(norm * plm * cos(absm * phi) * (-1)^absm)
    end
end

function legendre_p(l::Int, m::Int, x::T) where {T <: AbstractFloat}
    pmm = one(T)
    if m > 0
        somx2 = sqrt((one(T) - x) * (one(T) + x))
        fact  = one(T)
        for _ in 1:m
            pmm  *= -fact * somx2
            fact += T(2)
        end
    end
    m == l && return pmm
    pmmp1 = x * (2m + 1) * pmm
    m + 1 == l && return pmmp1
    pll = zero(T)
    for ll in (m+2):l
        pll   = (x * (2ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm   = pmmp1
        pmmp1 = pll
    end
    return pll
end
macro with_precision(digits, expr)
    bits = ceil(Int, digits * log2(10))
    return quote
        setprecision(BigFloat, $bits) do
            $(esc(expr))
        end
    end
end