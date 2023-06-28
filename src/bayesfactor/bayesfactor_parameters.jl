using DataFrames


function bayesfactor_parameters(posterior; prior = nothing, direction = "two-sided", null = 0, verbose = true, args...)
    # Method body here...
end

function bayesfactor_pointnull(posterior; prior = nothing, direction = "two-sided", null = 0, verbose = true, args...)
    if length(null) > 1 && verbose
        println("'null' is a range - computing a ROPE based Bayes factor.")
    end

    bayesfactor_parameters(posterior; prior = prior, direction = direction, null = null, verbose = verbose, args...)
end

function bayesfactor_rope(posterior; prior = nothing, direction = "two-sided", null = rope_range(posterior), verbose = true, args...)
    if length(null) < 2 && verbose
        println("'null' is a point - computing a Savage-Dickey (point null) Bayes factor.")
    end

    bayesfactor_parameters(posterior; prior = prior, direction = direction, null = null, verbose = verbose, args...)
end


# Define types for posterior
abstract type Posterior end

struct NumericPosterior <: Posterior
    data::Number
end

struct StanRegPosterior <: Posterior
    data::Any
end

# Define functions
bf_parameters(p::Posterior; kwargs...) = bayesfactor_parameters(p; kwargs...)

bf_pointnull(p::Posterior; kwargs...) = bayesfactor_pointnull(p; kwargs...)

bf_rope(p::Posterior; kwargs...) = bayesfactor_rope(p; kwargs...)

function bayesfactor_parameters(p::NumericPosterior; prior = nothing, direction = "two-sided", null = 0, verbose = true, kwargs...)
    if isnothing(prior)
        prior = p
        if verbose
            @warn "Prior not specified! Please specify a prior (in the form 'prior = distribution_normal(1000, 0, 1)') to get meaningful results."
        end
    end
    prior = DataFrame(X = prior)
    posterior = DataFrame(X = p)
    
    # Here, we need to define bayesfactor_parameters_data_frame which is not present in the R code you provided
    # Assuming it returns a DataFrame
    sdbf = bayesfactor_parameters_data_frame(posterior = posterior, prior = prior, direction = direction, null = null, verbose = verbose, kwargs...)
    delete!(sdbf, :Parameter)
    return sdbf
end


function bayesfactor_parameters(p::StanRegPosterior; prior = nothing, direction = "two-sided", null = 0, verbose = true, effects = ["fixed", "random", "all"], component = ["conditional", "location", "smooth_terms", "sigma", "zi", "zero_inflated", "all"], parameters = nothing, kwargs...)
    
    # The conversion of this function requires the conversion of the 'insight::clean_parameters' function
    # 'insight::clean_parameters' appears to be a function from the R 'insight' package
    cleaned_parameters = clean_parameters(p.data)  # assuming clean_parameters is already defined in Julia
    
    # 'match.arg' in R can be substituted by checking if the input is within the list of choices in Julia
    if !(effects in ["fixed", "random", "all"])
        error("Invalid 'effects' argument. Choices are 'fixed', 'random', or 'all'.")
    end

    if !(component in ["conditional", "location", "smooth_terms", "sigma", "zi", "zero_inflated", "all"])
        error("Invalid 'component' argument. Choices are 'conditional', 'location', 'smooth_terms', 'sigma', 'zi', 'zero_inflated', or 'all'.")
    end

    # We also need to define the '.clean_priors_and_posteriors' function in Julia
    samps = clean_priors_and_posteriors(p.data, prior, verbose = verbose, effects = effects, component = component, parameters = parameters)
    
    # Here, we need to define 'bayesfactor_parameters_data_frame' which is not present in the R code you provided
    # Assuming it returns a DataFrame
    temp = bayesfactor_parameters_data_frame(posterior = samps.posterior, prior = samps.prior, direction = direction, null = null, verbose = verbose, kwargs...)
    
    # '.prepare_output' function also needs to be defined in Julia
    # Also, 'inherits' function in R has no direct equivalent in Julia and has to be treated specifically depending on context
    bf_val = prepare_output(temp, cleaned_parameters, isa(p.data, StanMVReg))  # assuming StanMVReg is already defined in Julia
    
    # Similar to 'class' in R, you can use typeof in Julia to get the type of a variable
    # But unlike R, you cannot directly change the type of a variable in Julia
    # Instead, you can create a new struct that has the properties you want
    
    bf_val.clean_parameters = cleaned_parameters
    bf_val.hypothesis = temp.hypothesis
    bf_val.direction = temp.direction
    bf_val.plot_data = temp.plot_data

    return bf_val
end

#abstract type BrmsFitPosterior end
abstract type BlavaanPosterior end
abstract type EmmGridPosterior end

struct StanRegPosterior <: Posterior
    data::Any
end

#bayesfactor_parameters(p::BrmsFitPosterior; kwargs...) = bayesfactor_parameters(StanRegPosterior(p.data); kwargs...)

function bayesfactor_parameters(p::BlavaanPosterior; prior = nothing, direction = "two-sided", null = 0, verbose = true, kwargs...)
    
    cleaned_parameters = clean_parameters(p.data)  # assuming clean_parameters is already defined in Julia
    
    samps = clean_priors_and_posteriors(p.data, prior, verbose = verbose)
    
    temp = bayesfactor_parameters_data_frame(posterior = samps.posterior, prior = samps.prior, direction = direction, null = null, verbose = verbose, kwargs...)
    
    bf_val = prepare_output(temp, cleaned_parameters)
    
    bf_val.clean_parameters = cleaned_parameters
    bf_val.hypothesis = temp.hypothesis
    bf_val.direction = temp.direction
    bf_val.plot_data = temp.plot_data

    return bf_val
end

function bayesfactor_parameters(p::EmmGridPosterior; prior = nothing, direction = "two-sided", null = 0, verbose = true, kwargs...)
    
    samps = clean_priors_and_posteriors(p.data, prior, verbose = verbose)
    
    return bayesfactor_parameters_data_frame(posterior = samps.posterior, prior = samps.prior, direction = direction, null = null, verbose = verbose, kwargs...)
end


struct DataFramePosterior
    data::DataFrame
end

struct EmmListPosterior
    data::Any
end

bayesfactor_parameters(p::EmmListPosterior; kwargs...) = bayesfactor_parameters(DataFramePosterior(p.data); kwargs...)

function bayesfactor_parameters(p::DataFramePosterior; prior = nothing, direction = "two-sided", null = 0, verbose = true, kwargs...)
    
    direction = get_direction(direction)
    
    if isnothing(prior)
        prior = p.data
        if verbose
            println("Prior not specified! Please specify priors (with column order matching 'posterior') to get meaningful results.")
        end
    end

    if verbose && length(null) == 1 && (size(p.data, 1) < 40000 || size(prior, 1) < 40000)
        println("Bayes factors might not be precise. For precise Bayes factors, sampling at least 40,000 posterior samples is recommended.")
    end
    
    sdbf = zeros(size(p.data, 2))
    for par in 1:size(p.data, 2)
        sdbf[par] = bayesfactor_parameters(p.data[:, par], prior[:, par], direction = direction, null = null, kwargs...)
    end
    
    bf_val = DataFrame(
        Parameter = names(p.data),
        log_BF = log.(sdbf)
    )

    bf_val.hypothesis = null
    bf_val.direction = direction
    bf_val.plot_data = make_BF_plot_data(p.data, prior, direction, null, kwargs...)
    
    return bf_val
end


struct DrawsPosterior
    data::Any
end

struct RVarPosterior
    data::Any
end

bayesfactor_parameters(p::RVarPosterior; kwargs...) = bayesfactor_parameters(DrawsPosterior(p.data); kwargs...)

function bayesfactor_parameters(p::DrawsPosterior; prior = nothing, direction = "two-sided", null = 0, verbose = true, kwargs...)
    return bayesfactor_parameters(posterior_draws_to_df(p.data), prior = prior, direction = direction, null = null, verbose = verbose, kwargs...)
end

function bayesfactor_parameters(posterior, prior, direction = 0, null = 0, kwargs...)
    @assert length(null) ∈ [1, 2]

    if all(posterior .== prior)
        return 1
    end

    # check_if_installed("logspline")

    if length(null) == 1
        function relative_density(samples)
            f_samples = logspline(samples, kwargs...)
            d_samples = dlogspline(null, f_samples)

            if direction < 0
                norm_samples = plogspline(null, f_samples)
            elseif direction > 0
                norm_samples = 1 - plogspline(null, f_samples)
            else
                norm_samples = 1
            end

            return d_samples / norm_samples
        end

        return relative_density(prior) / relative_density(posterior)
    elseif length(null) == 2
        null = sort(null)
        null[isinf.(null)] = 1.797693e+308 .* sign.(null[isinf.(null)])

        f_prior = logspline(prior, kwargs...)
        f_posterior = logspline(posterior, kwargs...)

        h0_prior = diff(plogspline(null, f_prior))
        h0_post = diff(plogspline(null, f_posterior))

        BF_null_full = h0_post / h0_prior

        if direction < 0
            h1_prior = plogspline(minimum(null), f_prior)
            h1_post = plogspline(minimum(null), f_posterior)
        elseif direction > 0
            h1_prior = 1 - plogspline(maximum(null), f_prior)
            h1_post = 1 - plogspline(maximum(null), f_posterior)
        else
            h1_prior = 1 - h0_prior
            h1_post = 1 - h0_post
        end
        BF_alt_full = h1_post / h1_prior

        return BF_alt_full / BF_null_full
    end
end


struct BayesfactorModels
    data::Any
end

struct Sim
    data::Any
end

struct SimMerMod
    data::Any
end

bayesfactor_parameters(::BayesfactorModels) = error("""
    Oh no, 'bayesfactor_parameters()' does not know how to deal with multiple models :(
    You might want to use 'bayesfactor_inclusion()' here to test specific terms across models.
    """)

bayesfactor_parameters(::Sim) = error("""
    Bayes factors are based on the shift from a prior to a posterior.
    Since simulated draws are not based on any priors, computing Bayes factors does not make sense :(
    You might want to try `rope`, `ci`, `pd` or `pmap` for posterior-based inference.
    """)

bayesfactor_parameters(::SimMerMod) = bayesfactor_parameters(Sim())
