using DataFrames

# Base check_prior function
function check_prior(model; method="gelman", simulate_priors=true, kwargs...)
    return dispatch_check_prior(model; method=method, simulate_priors=simulate_priors, kwargs...)
end

# Method specific to brmsfit model
function check_prior(model::Brmsfit; method="gelman", simulate_priors=true, effects=["fixed", "random", "all"],
                     component=["conditional", "zi", "zero_inflated", "all"], parameters=nothing, verbose=true, kwargs...)
    effects = get_matching_arg(effects)
    component = get_matching_arg(component)

    posteriors = get_parameters(model; effects=effects, component=component, parameters=parameters)

    if simulate_priors
        priors = simulate_prior(model; effects=effects, component=component, parameters=parameters, verbose=verbose)
    else
        priors = get_parameters(unupdate(model; verbose=false); effects=effects, component=component, parameters=parameters)
    end

    return internal_check_prior(priors, posteriors; method=method, verbose=verbose, cleaned_parameters=clean_parameters(model))
end

# Assign the same function to different model types
const check_prior(::Stanreg) = check_prior(::Brmsfit)
const check_prior(::Blavaan) = check_prior(::Brmsfit)

function internal_check_prior(priors, posteriors; method="gelman", verbose=true, cleaned_parameters=nothing)
    
    # Sanity check for matching parameters.
    if !isnothing(cleaned_parameters) && size(priors, 2) != size(posteriors, 2)
        if "Effects" in names(cleaned_parameters)
            cleaned_parameters = cleaned_parameters[cleaned_parameters.Effects .== "fixed", :]
        end

        # rename cleaned parameters, so they match name of prior parameter column
        cp = cleaned_parameters[:, :Cleaned_Parameter]
        cp = replace.(cp, r"(.*)(\.|\[)\d+(\.|\])" => s"\1")
        cp[cp .== "Intercept"] .= "(Intercept)"
        cleaned_parameters[:, :Cleaned_Parameter] = cp
        rename!(priors, "Intercept" => "(Intercept)")

        # Ensuring that ncol of priors is the same as ncol of posteriors
        if size(posteriors, 2) > size(priors, 2)
            matched_columns = filter(!isnothing, [findfirst(==(p), names(cleaned_parameters[:, :Cleaned_Parameter])) for p in names(priors)])
            priors = priors[:, matched_columns]
        else
            matched_columns = filter(!isnothing, [findfirst(==(p), names(priors)) for p in cleaned_parameters[:, :Cleaned_Parameter]])
            priors = priors[:, matched_columns]
        end
        rename!(priors, names(priors) .=> cleaned_parameters[:, :Parameter][matched_columns])
    end
    # still different ncols?
    if size(priors, 2) != size(posteriors, 2)
        common_columns = intersect(names(priors), names(posteriors))
        priors = priors[:, common_columns]
        posteriors = posteriors[:, common_columns]
        if verbose
            @warn("Parameters and priors could not be fully matched. Only returning results for parameters with matching priors.")
        end
    end

    # For priors whose distribution cannot be simulated, prior values are all NA. Catch those, and warn user
    all_missing = mapslices(x -> all(isnan, x), priors, dims=1)
    if any(all_missing) && verbose
        @warn("Some priors could not be simulated.")
    end

    gelman(prior, posterior) = all(isnan, prior) ? "not determinable" : std(posterior, skipmissing=true) > 0.1 * std(prior, skipmissing=true) ? "informative" : "uninformative"

    lakeland(prior, posterior) = begin
        # "hdi" and "rope" functions need to be implemented in Julia as they are not directly available. 
        # The actual implementation depends on the specific requirements of your model and data.
        # Placeholder functions are used here as a placeholder, you should replace them with the correct code.
        if all(isnan, prior)
            "not determinable"
        else
            hdi = hdi(prior, ci=0.95)
            r = rope(posterior, ci=1, range=hdi)
            if r > 0.99
                "informative"
            else
                "misinformative"
            end
        end
    end

    if method == "gelman"
        result = map(gelman, eachcol(priors), eachcol(posteriors))
    elseif method == "lakeland"
        result = map(lakeland, eachcol(priors), eachcol(posteriors))
    else
        @error("Method should be 'gelman' or 'lakeland'.")
    end

    result_df = DataFrame(Parameter = names(posteriors), Prior_Quality = result)
    return result_df
end
