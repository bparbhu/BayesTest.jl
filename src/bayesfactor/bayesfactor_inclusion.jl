using DataFrames, Statistics

function bayesfactor_inclusion(models::DataFrame; match_models=false, prior_odds=nothing)
    
    if models.unsupported_models
        throw(ArgumentError("Cannot compute inclusion Bayes factors - passed models are not (yet) supported."))
    end

    df_model = get_model_table(models, priorOdds = prior_odds)
    effnames = names(df_model)[4:end]

    if match_models
        effects_matrix = Matrix(df_model[:, 4:end])

        df_interaction = DataFrame(effnames = effnames)

        for eff in effnames
            df_interaction[!, eff] = [includes_interaction(e, effnames) for e in effnames]
        end

        df_interaction = Matrix(df_interaction[:, 2:end])
    end

    df_effect = DataFrame(
        effnames = effnames,
        Pinc = fill(missing, length(effnames)),
        PincD = fill(missing, length(effnames)),
        log_BF = fill(missing, length(effnames))
    )

    for eff in effnames
        if match_models
            idx1 = df_interaction[:, eff]
            idx2 = df_interaction[eff, :]

            has_not_high_order_interactions = .!any(effects_matrix[:, idx1], dims=1)

            ind_include = has_not_high_order_interactions .& effects_matrix[:, eff]

            ind_exclude = all(effects_matrix[:, idx2], dims=1) .&
                has_not_high_order_interactions .&
                .!effects_matrix[:, eff]

            df_model_temp = df_model[ind_include .| ind_exclude, :]
        else
            df_model_temp = df_model
        end

        mwith = findall(x->x==eff, df_model_temp)
        mwithprior = sum(df_model_temp[mwith, :priorProbs])
        mwithpost = sum(df_model_temp[mwith, :postProbs])

        mwithoutprior = sum(df_model_temp[setdiff(1:end, mwith), :priorProbs])
        mwithoutpost = sum(df_model_temp[setdiff(1:end, mwith), :postProbs])

        df_effect[df_effect.effnames .== eff, :Pinc] .= mwithprior
        df_effect[df_effect.effnames .== eff, :PincD] .= mwithpost
        df_effect[df_effect.effnames .== eff, :log_BF] .= log(mwithpost/mwithoutpost) - log(mwithprior/mwithoutprior)
    end

    df_effect = df_effect[:, 2:end]
    rename!(df_effect, [:p_prior, :p_posterior, :log_BF])
    return df_effect
end

# Here are placeholder functions. You need to replace them with your own functions.
#get_model_table(models, prior_odds) = DataFrame()
#includes_interaction(eff, effnames) = true


function bayesfactor_inclusion(models, match_models=false, prior_odds=nothing)
    models = bayesfactor_models(models)
    return bayesfactor_inclusion(models, match_models = match_models, prior_odds = prior_odds)
end

function includes_interaction(eff, effnames)
    eff_b = split(eff, ":")
    effnames_b = [split(effname, ":") for effname in effnames]

    is_int = [length(x) > 1 for x in effnames_b]

    temp = fill(false, length(effnames))

    for rr in eachindex(effnames)
        if is_int[rr]
            temp[rr] = all(x -> x in effnames_b[rr], eff_b) && !all(x -> x in eff_b, effnames_b[rr])
        end
    end

    return temp
end

# Placeholder function, you need to replace it with your own function.
#bayesfactor_models(models) = models
