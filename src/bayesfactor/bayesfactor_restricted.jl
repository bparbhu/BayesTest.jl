using DataFrames
using StatsBase


function bayesfactor_restricted(posterior, hypothesis; prior=nothing, verbose=true, kwargs...)
    if typeof(posterior) == StanReg
        return bayesfactor_restricted(posterior, hypothesis, prior; verbose=verbose, kwargs...)
    else
        error("Unknown method for type ", typeof(posterior))
    end
end

bf_restricted = bayesfactor_restricted

struct StanReg
    data::Any
end

function bayesfactor_restricted(posterior::StanReg, hypothesis, prior; 
                                 verbose=true, effects=["fixed", "random", "all"], 
                                 component=["conditional", "zi", "zero_inflated", "all"], 
                                 kwargs...)

    effects = get_arg(effects)
    component = get_arg(component)

    samps = clean_priors_and_posteriors(posterior, prior, effects, component, verbose=verbose)

    # Get savage-dickey BFs
    return bayesfactor_restricted(samps["posterior"], samps["prior"], hypothesis=hypothesis)
end


# Define placeholder structs
struct Brmsfit
    data::Any
end

struct Blavaan
    data::Any
end

struct EmmGrid
    data::Any
end

struct EmmList
    data::Any
end

bayesfactor_restricted(posterior::Brmsfit, hypothesis, prior; kwargs...) = 
    bayesfactor_restricted(posterior::StanReg, hypothesis, prior; kwargs...)

function bayesfactor_restricted(posterior::Blavaan, hypothesis, prior; verbose=true, kwargs...)
    samps = clean_priors_and_posteriors(posterior, prior, verbose=verbose)

    # Get savage-dickey BFs
    return bayesfactor_restricted(samps["posterior"], samps["prior"], hypothesis=hypothesis)
end

function bayesfactor_restricted(posterior::EmmGrid, hypothesis, prior; verbose=true, kwargs...)
    samps = clean_priors_and_posteriors(posterior, prior, verbose=verbose)

    return bayesfactor_restricted(samps["posterior"], samps["prior"], hypothesis=hypothesis)
end

bayesfactor_restricted(posterior::EmmList, hypothesis, prior; kwargs...) = 
    bayesfactor_restricted(posterior::EmmGrid, hypothesis, prior; kwargs...)


function test_hypothesis(x, data)
    try
        x_logical = eval(Meta.parse(x), data)
    catch e
        cnames = names(data)
        is_name = Symbol.(cnames) .== cnames
        cnames[.!is_name] = "`".*cnames[.!is_name].*"`"
        error(e, " Available parameters are: ", join(cnames, ", "))
    end
    if (!all(x -> isa(x, Bool), x_logical))
        error("Hypotheses must be logical.")
    end
    return x_logical
end

function bayesfactor_restricted(posterior::DataFrame, hypothesis, prior::Union{DataFrame,Nothing}=nothing)
    if isnothing(prior)
        prior = posterior
        println("Prior not specified! Please specify priors (with column names matching 'posterior') to get meaningful results.")
    end

    p_hypothesis = Meta.parse.(hypothesis)
    posterior_l = DataFrame([test_hypothesis(x, posterior) for x in p_hypothesis], Symbol.(hypothesis))
    prior_l = DataFrame([test_hypothesis(x, prior) for x in p_hypothesis], Symbol.(hypothesis))

    posterior_p = mean.(eachcol(posterior_l))
    prior_p = mean.(eachcol(prior_l))
    BF = posterior_p ./ prior_p

    res = DataFrame(
        Hypothesis = hypothesis,
        p_prior = prior_p,
        p_posterior = posterior_p,
        log_BF = log.(BF)
    )

    res.bool_results = (posterior = posterior_l, prior = prior_l)
    res
end

function bayesfactor_restricted(posterior::Draws, hypothesis, prior::Union{Draws, Nothing}=nothing)
    bayesfactor_restricted(posterior_draws_to_df(posterior), hypothesis, prior)
end

bayesfactor_restricted(posterior::Rvar, hypothesis, prior::Union{Rvar, Nothing}=nothing) = bayesfactor_restricted(posterior::Draws, hypothesis, prior)

function as_logical(x::DataFrame, which::String)
    return convert(Matrix, x.bool_results[Symbol(which)])
end


