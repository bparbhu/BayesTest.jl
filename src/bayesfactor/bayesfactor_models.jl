using DataFrames
using StatsBase
using BridgeSampling

function bayesfactor_models(;kwargs...)
    models = Dict(kwargs)
    denominator = get(models, "denominator", 1)
    verbose = get(models, "verbose", true)
    delete!(models, "denominator")
    delete!(models, "verbose")

    estimator = get(models, "estimator", "ML")
    check_response = get(models, "check_response", false)
    delete!(models, "estimator")
    delete!(models, "check_response")

    cleaned_models = cleanup_BF_models(models, denominator)
    model_forms = keys(cleaned_models)
    denominator = get(cleaned_models, "denominator")

    supported_models = [is_model_supported(m) for m in values(models)]
    if all(supported_models)
        temp_forms = [find_full_formula(m) for m in values(models)]
        has_terms = [length(f) > 0 for f in temp_forms]

        model_forms = [has_terms[i] ? temp_forms[i] : model_forms[i] for i in 1:length(model_forms)]
        supported_models = [!has_terms[i] ? false : supported_models[i] for i in 1:length(supported_models)]
    end

    try
        objects = ellipsis_info(models, verbose = false)
        were_checked = isa(objects, ListModels)

        if were_checked && verbose && !get(objects, "same_response")
            @warn "When comparing models, please note that probably not all models were fit from same data."
        end

        if were_checked && estimator == "REML" && any([is_mixed_model(m) for m in values(models)]) && !get(objects, "same_fixef") && verbose
            @warn "Information criteria (like BIC) based on REML fits (i.e. `estimator=\"REML\"`) are not recommended for comparison between models with different fixed effects. Consider setting `estimator=\"ML\"`."
        end
    catch
        @warn "Unable to validate that all models were fit with the same data."
    end

    try
        mBIC = [get_loglikelihood(m, estimator = estimator, check_response = check_response) |> BIC for m in values(models)]
    catch
        mBIC = [BIC(m) for m in values(models)]
    end

    mBFs = bic_to_bf(mBIC, denominator = mBIC[denominator], log = true)

    res = DataFrame(Model = model_forms, log_BF = mBFs)

    return bf_models_output(res, denominator = denominator, bf_method = "BIC approximation", unsupported_models = !all(supported_models), model_names = keys(models))
end

# Placeholder functions, replace these with your own
# cleanup_BF_models(models, denominator) = models
# is_model_supported(m) = true
# find_full_formula(m) = ""
# ellipsis_info(models; kwargs...) = Dict()
# is_mixed_model(m) = false
# get_loglikelihood(m; kwargs...) = rand()
# bic_to_bf(mBIC; kwargs...) = mBIC
# bf_models_output(res; kwargs...) = res


function bayesfactor_models_stan(mods; denominator = 1, verbose = true)
    n_samps = [((get(find_algorithm(x), "iterations", get(find_algorithm(x), "sample")) - get(find_algorithm(x), "warmup")) * get(find_algorithm(x), "chains")) for x in values(mods)]
    if any(n_samps .< 4e4) && verbose
        @warn "Bayes factors might not be precise. For precise Bayes factors, sampling at least 40,000 posterior samples is recommended."
    end

    if typeof(first(values(mods))) == Blavaan
        res = bayesfactor_models_stan_SEM(mods, denominator = denominator, verbose = verbose)
        bf_method = "marginal likelihoods (Laplace approximation)"
        unsupported_models = true
    else
        res = bayesfactor_models_stan_REG(mods, denominator = denominator, verbose = verbose)
        bf_method = "marginal likelihoods (bridgesampling)"
        unsupported_models = false
    end

    return bf_models_output(res, denominator = denominator, bf_method = bf_method, unsupported_models = unsupported_models)
end

function bayesfactor_models_stan_REG(mods; denominator = 1, verbose = true)
    check_if_installed("bridgesampling")

    resps = [get_response(x) for x in values(mods)]
    from_same_data_as_den = [x == resps[denominator] for x in resps[1:denominator-1]]

    if !all(from_same_data_as_den)
        error("Models were not computed from the same data.")
    end

    mML = [get_marglik(x, verbose = verbose) for x in values(mods)]
    mBFs = [bf(x, mML[denominator], log = true)["bf"] for x in mML]
    mforms = [find_full_formula(x) for x in values(mods)]

    return DataFrame(Model = mforms, log_BF = mBFs)
end

function bayesfactor_models_stan_SEM(mods; denominator = 1, verbose = true)
    mBFs = [blavCompare(x, mods[denominator])["bf"][1] for x in values(mods)]
    return DataFrame(Model = keys(mods), log_BF = mBFs)
end

function bayesfactor_models_stanreg(;kwargs...)
    check_if_installed("rstanarm")

    mods = Dict(kwargs)
    denominator = get(mods, "denominator", 1)
    delete!(mods, "denominator")

    cleaned_models = cleanup_BF_models(mods, denominator)
    denominator = get(cleaned_models, "denominator")

    return bayesfactor_models_stan(cleaned_models, denominator = denominator, verbose = verbose)
end

# Placeholder functions, replace these with your own
# find_algorithm(x) = Dict("iterations" => 500, "warmup" => 100, "chains" => 4)
# get_response(x) = "response"
# check_if_installed(package) = println("Checking if $package is installed...")
# get_marglik(model; verbose = true) = Dict("bf" => 1.0)
# bf(x, y; log = true) = Dict("bf" => 1.0)
# find_full_formula(model) = "formula"
# blavCompare(x, y) = Dict("bf" => [1.0])
# cleanup_BF_models(mods, denominator) = Dict("denominator" => denominator)

function bayesfactor_models_blavaan(;kwargs...)
    check_if_installed("blavaan")

    mods = Dict(kwargs)
    denominator = get(mods, "denominator", 1)
    delete!(mods, "denominator")

    cleaned_models = cleanup_BF_models(mods, denominator)
    denominator = get(cleaned_models, "denominator")

    return bayesfactor_models_stan(cleaned_models, denominator = denominator, verbose = verbose)
end

function bayesfactor_models_BFBayesFactor(;kwargs...)
    check_if_installed("BayesFactor")

    models = Dict(kwargs)
    mBFs = [0; extractBF(values(models), true, true)]
    mforms = [get(x, "shortName") for x in [get(models, "denominator"), get(models, "numerator")]]

    if !(typeof(get(models, "denominator")) <: BFlinearModel)
        mforms = clean_non_linBF_mods(mforms)
    else
        mforms[mforms .== "Intercept only"] .= "1"
    end

    res = DataFrame(Model = mforms, log_BF = mBFs)

    return bf_models_output(res, denominator = 1, bf_method = "JZS (BayesFactor)", unsupported_models = !(typeof(get(models, "denominator")) <: BFlinearModel))
end

# Placeholder functions, replace these with your own
# check_if_installed(package) = println("Checking if $package is installed...")
# extractBF(models, arg1, arg2) = [1.0 for model in models]
# cleanup_BF_models(mods, denominator) = Dict("denominator" => denominator)
# clean_non_linBF_mods(mforms) = mforms


function update_bayesfactor_models(object, subset = nothing, reference = nothing)
    if !isnothing(reference)
        if reference == "top"
            reference = argmax(object.log_BF)
        elseif reference == "bottom"
            reference = argmin(object.log_BF)
        end
        object.log_BF .-= object.log_BF[reference]
        object[!, "denominator"] = reference
    end

    denominator = object[!, "denominator"]

    if !isnothing(subset)
        if all(subset .< 0)
            subset = 1:nrow(object)[subset]
        end
        object_subset = object[subset, :]

        if denominator in subset
            object_subset[!, "denominator"] = findfirst(==(denominator), subset)
        else
            object_subset = vcat(object[denominator, :], object_subset)
            object_subset[!, "denominator"] = 1
        end
        object = object_subset
    end
    object
end

function as_matrix_bayesfactor_models(x)
    out = -broadcast(-, x.log_BF, x.log_BF')
    out = DataFrame(out, Symbol.(x.Model))

    out[!, "bayesfactor_models_matrix"] = out
    out
end

function cleanup_BF_models(mods, denominator)
    if length(mods) == 1 && typeof(mods[1]) <: Dict
        mods = mods[1]
        try
            mod_names = safe_deparse.(keys(mods))
        catch e
            mod_names = nothing
        end
        if !isnothing(mod_names) && length(mod_names) == length(mods)
            mods = Dict(mod_names[i] => v for (i, v) in enumerate(values(mods)))
        end
    end

    if !isa(denominator[1], Number)
        denominator_model = findfirst(==(names(denominator)), keys(mods))

        if isnothing(denominator_model)
            mods = Dict(keys(mods) .=> values(mods), names(denominator) .=> denominator)
            denominator = length(mods)
        else
            denominator = denominator_model
        end
    else
        denominator = denominator[1]
    end

    mods["denominator"] = denominator
    mods
end

# Placeholder function, replace this with your own
# safe_deparse(item) = string(item)

function bf_models_output(res, denominator = 1, bf_method = "method", unsupported_models = false, model_names = nothing)
    res[:, "denominator"] = denominator
    res[:, "BF_method"] = bf_method
    res[:, "unsupported_models"] = unsupported_models
    res[:, "model_names"] = model_names
    res[:, "class"] = ["bayesfactor_models", "see_bayesfactor_models", typeof(res)]
    return res
end

function find_full_formula(mod)
    formulas = find_formula(mod) # Assume find_formula is defined and works similar to R's insight::find_formula
    
    conditional = nothing
    random = nothing
    if !isnothing(formulas.conditional)
        conditional = split(formulas.conditional, " ")[3]
    end

    if !isnothing(formulas.random)
        if !isa(formulas.random, Vector)
            formulas.random = [formulas.random]
        end
        random = join(["(" * split(r, " ")[2] * ")" for r in formulas.random], " + ")
    end
    return join([conditional, random], " + ")
end


function clean_non_linBF_mods(m_names)
    try
        m_txt = Array{String}(undef, length(m_names))

        is_null = startswith.(m_names, "Null")
        is_rho = occursin.("rho", m_names)
        is_mu = occursin.("mu", m_names)
        is_d = occursin.("d", m_names)
        is_p = occursin.("p", m_names)
        is_range = occursin.("<", m_names)

        m_txt[.!is_null .& is_range] = replace(m_names[.!is_null .& is_range], r"^[^\s]*\s[^\s]*\s"=>"")
        if any(is_d .& is_p)
            is_null .= .!startswith.(m_names, "Non")
            temp = m_names[is_null][1]
            aa = match(r"\(.*\)", temp).match

            m_txt[is_null] = replace(aa, "a="=>"a = ")
            m_txt[.!is_null .& .!is_range] = replace(aa, "a="=>"a != ")
        elseif any(is_rho)
            m_txt[is_null] .= "rho = 0"
            m_txt[.!is_null .& .!is_range] .= "rho != 0"
            m_txt .= replace(m_txt, "<rho<"=>" < rho < ")
        elseif any(is_d .| is_mu)
            m_txt[is_null] .= "d = 0"
            m_txt[.!is_null .& .!is_range] .= "d != 0"
            m_txt .= replace(m_txt, "<d<"=>" < d < ")
        elseif any(is_p)
            temp = m_names[is_null][1]
            pp = match(r"[0-9|\.]+", temp).match

            m_txt[is_null] .= "p = $pp"
            m_txt[.!is_null .& .!is_range] .= "p != $pp"
            m_txt .= replace(m_txt, "<p<"=>" < p < ")
        else
            error("!")
        end

        is_wrapped = occursin.("(", m_txt)
        m_txt[.!is_wrapped] .= "($m_txt[.!is_wrapped])"

        return m_txt
    catch e
        return m_names
    end
end

function get_marglik(mod, verbose; kwargs...)
    if typeof(mod) == DataFrame && "marglik" in names(mod)
        return median(mod.marglik.logml)
    end
    if verbose
        println("Computation of Marginal Likelihood: estimating marginal likelihood, please wait...")
    end
    bridge_sampler(mod, silent=true; kwargs...)
end

using Requires

function bayesfactor_models(args::Vector{T}; denominator=1, verbose=true) where T<:Brmsfit
    @require Brms "No Brms package found. Please install Brms to use this function."

    # Organize the models and their names
    mods = args
    denominator = [denominator]

    # Extract names
    mods_names = [string(arg) for arg in args]
    denominator_name = string(denominator)

    # 'cleanup_BF_models', 'bayesfactor_models_stan' need to be defined in Julia
    mods = cleanup_BF_models(mods, denominator)
    denominator = get(denominator, "denominator", true)

    return bayesfactor_models_stan(mods, denominator=denominator, verbose=verbose)
end
