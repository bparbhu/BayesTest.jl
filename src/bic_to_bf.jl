function bic_to_bf(bic, denominator; log=false)
    delta = (denominator - bic) / 2
    if log
        return delta
    else
        return exp(delta)
    end
end
