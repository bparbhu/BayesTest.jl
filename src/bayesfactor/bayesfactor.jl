function bayesfactor(;mods = nothing,
                      prior = nothing,
                      direction = "two-sided",
                      null = 0,
                      hypothesis = nothing,
                      effects = "fixed",
                      verbose = true,
                      denominator = 1,
                      match_models = false,
                      prior_odds = nothing)
    
    if mods != nothing
        if length(mods) > 1
            return bayesfactor_models(mods..., denominator = denominator)
        elseif typeof(mods[1]) <: bayesfactor_models
            return bayesfactor_inclusion(mods..., match_models = match_models, prior_odds = prior_odds)
        elseif typeof(mods[1]) <: BFBayesFactor
            if typeof(mods[1].numerator[1]) <: BFlinearModel
                return bayesfactor_inclusion(mods..., match_models = match_models, prior_odds = prior_odds)
            else
                return bayesfactor_models(mods...)
            end
        elseif hypothesis != nothing
            return bayesfactor_restricted(mods..., prior = prior, verbose = verbose, effects = effects)
        else
            return bayesfactor_parameters(mods..., prior = prior, direction = direction, null = null, effects = effects, verbose = verbose)
        end
    end
end