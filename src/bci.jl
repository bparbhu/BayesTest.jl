using DataFrames

abstract type BCI end

# Define your helper functions here
function bci_helper(data, ci, verbose)
    # Your implementation here
end

function compute_interval_dataframe(data, ci, verbose, func)
    # Your implementation here
end

function bci(x::BCI; kwargs...)
    throw(MethodError(bci, (x,)))
end

const bcai = bci

struct Numeric <: BCI
    data::Vector{Float64}
end

function bci(x::Numeric; ci=0.95, verbose=true, kwargs...)
    out = hcat(ci, [bci_helper(x.data, c, verbose=verbose) for c in ci])
    out
end

struct DataFrame <: BCI
    data::Dict{String, Vector{Float64}}
end

function bci(x::DataFrame; ci=0.95, verbose=true, kwargs...)
    dat = compute_interval_dataframe(x.data, ci, verbose, "bci")
    dat
end

struct MCMCglmm <: BCI
    Sol::Matrix{Float64}
    Fixed::Dict{Symbol, Any}
end

function bci(x::MCMCglmm; ci=0.95, verbose=true, kwargs...)
    nF = x.Fixed[:nfl]
    d = Dict([(:col$i, vec(x.Sol[:, i])) for i in 1:nF])
    dat = compute_interval_dataframe(d, ci, verbose, "bci")
    dat
end

struct Bamlss <: BCI
    data::Dict{String, Vector{Float64}}
end

function bci(x::Bamlss; ci=0.95, component="all", verbose=true, kwargs...)
    d = get_parameters(x.data, component)
    dat = compute_interval_dataframe(d, ci, verbose, "bci")
    dat
end


# Add the new types here
struct Bcplm <: BCI
    data::Dict{String, Vector{Float64}}
end

struct SimMerMod <: BCI
    data::Any  # replace Any with the actual type of the data
    effects::String
    parameters::Any  # replace Any with the actual type of the parameters
end

struct Sim <: BCI
    data::Any  # replace Any with the actual type of the data
    parameters::Any  # replace Any with the actual type of the parameters
end

bci(x::Bcplm; ci=0.95, verbose=true, kwargs...) = bci_bcplm(x, ci, verbose)

# Since all these types have the same implementation as Bcplm, we can reuse the function.
bci(x::BayesQR; kwargs...) = bci(x::Bcplm; kwargs...)
bci(x::Blrm; kwargs...) = bci(x::Bcplm; kwargs...)
bci(x::McmcList; kwargs...) = bci(x::Bcplm; kwargs...)
bci(x::BGGM; kwargs...) = bci(x::Bcplm; kwargs...)

bci(x::SimMerMod; ci=0.95, verbose=true, kwargs...) = bci_simMerMod(x, ci, verbose)

bci(x::Sim; ci=0.95, verbose=true, kwargs...) = bci_sim(x, ci, verbose)


function bci_bcplm(x, ci, verbose)
    d = get_parameters(x.data)
    dat = compute_interval_dataframe(d, ci, verbose, "bci")
    dat
end

function bci_simMerMod(x, ci, verbose)
    dat = compute_interval_simMerMod(x.data, ci, x.effects, x.parameters, verbose, "bci")
    out = dat[:result]
    out
end

function bci_sim(x, ci, verbose)
    dat = compute_interval_sim(x.data, ci, x.parameters, verbose, "bci")
    out = dat[:result]
    out
end


# Add the new types here
struct EmmGrid <: BCI
    data::Any  # replace Any with the actual type of the data
end

struct Stanreg <: BCI
    data::Any  # replace Any with the actual type of the data
    effects::String
    component::String
    parameters::Any  # replace Any with the actual type of the parameters
end

struct Brmsfit <: BCI
    data::Any  # replace Any with the actual type of the data
    effects::String
    component::String
    parameters::Any  # replace Any with the actual type of the parameters
end

# Here's how you can define the conversion functions
bci(x::EmmGrid; ci=0.95, verbose=true, kwargs...) = bci_emmGrid(x, ci, verbose)
bci(x::EmmList; kwargs...) = bci(x::EmmGrid; kwargs...)
bci(x::Stanreg; ci=0.95, verbose=true, kwargs...) = bci_stanreg(x, ci, verbose)
bci(x::Stanfit; kwargs...) = bci(x::Stanreg; kwargs...)
bci(x::Blavaan; kwargs...) = bci(x::Stanreg; kwargs...)
bci(x::Brmsfit; ci=0.95, verbose=true, kwargs...) = bci_brmsfit(x, ci, verbose)

function bci_emmGrid(x, ci, verbose)
    xdf = get_parameters(x.data)
    dat = bci(xdf, ci=ci, verbose=verbose)
    dat
end

function bci_stanreg(x, ci, verbose)
    params = get_parameters(x.data, effects=x.effects, component=x.component, parameters=x.parameters)
    out = prepare_output(bci(params, ci=ci, verbose=verbose), clean_parameters(x.data))
    out
end

function bci_brmsfit(x, ci, verbose)
    params = get_parameters(x.data, effects=x.effects, component=x.component, parameters=x.parameters)
    out = prepare_output(bci(params, ci=ci, verbose=verbose), clean_parameters(x.data))
    out
end


struct BFBayesFactor <: BCI
    data::Any  # replace Any with the actual type of the data
end

struct GetPredicted <: BCI
    data::Any  # replace Any with the actual type of the data
end

bci(x::BFBayesFactor; ci=0.95, verbose=true, kwargs...) = bci_BFBayesFactor(x, ci, verbose)
bci(x::GetPredicted; kwargs...) = bci_get_predicted(x, kwargs...)

function bci_BFBayesFactor(x, ci, verbose)
    out = bci(get_parameters(x.data), ci=ci, verbose=verbose)
    out
end

function bci_get_predicted(x, kwargs)
    if "iterations" ∈ keys(x.data)
        out = bci(DataFrame(transpose(x.data["iterations"])), kwargs)
    else
        throw(DomainError("No iterations present in the output."))
    end
    out
end

function _bci(x, ci; verbose=true)
    check_ci = _check_ci_argument(x, ci, verbose)

    if !isnothing(check_ci)
        return check_ci
    end

    low = (1 - ci) / 2
    high = 1 - low
    sims = length(x)
    z_inv = length(x[x .< mean(x)]) / sims

    z = quantile(Normal(), z_inv)
    U = (sims - 1) .* (mean(x) - x)
    top = sum(U.^3)
    under = 6 * (sum(U.^2))^1.5
    a = top / under

    lower_inv = cdf(Normal(), z + (z + quantile(Normal(), low)) / (1 - a * (z + quantile(Normal(), low))))
    lower = quantile(x, lower_inv)
    upper_inv = cdf(Normal(), z + (z + quantile(Normal(), high)) / (1 - a * (z + quantile(Normal(), high))))
    upper = quantile(x, upper_inv)

    DataFrame(CI = ci, CI_low = lower, CI_high = upper)
end
