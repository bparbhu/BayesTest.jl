using Cubature, Interpolations

function area_under_curve(x, y; method="trapezoid", kwargs...)
    if length(x) != length(y)
        throw(ArgumentError("Length of x must be equal to length of y."))
    end

    idx = sortperm(x)
    x = x[idx]
    y = y[idx]

    if method == "trapezoid"
        return sum((y[1:end-1] .+ y[2:end]) ./ 2 .* diff(x))
    elseif method == "step"
        return sum(y[1:end-1] .* diff(x))
    elseif method == "spline"
        itp = interpolate((x,), y, Gridded(Linear()))
        (value, err) = hquadrature(itp, minimum(x), maximum(x))
        return value
    else
        throw(ArgumentError("Method must be one of: 'trapezoid', 'step', 'spline'."))
    end
end

const auc = area_under_curve
