struct ZILogPoisson{T} <: DiscreteUnivariateDistribution
  θ::T
  λ::T
end

function Distributions.logpdf(d::ZILogPoisson, y::Int)
  θ, λ = d.θ, d.λ
  if y == 0
    ll = [
      logpdf(BernoulliLogit(θ), 1),
      logpdf(BernoulliLogit(θ), 0) + logpdf(LogPoisson(λ), y)
    ]
    return logsumexp(ll)
  else
    ll = logpdf(BernoulliLogit(θ), 0) + logpdf(LogPoisson(λ), y)
    return ll
  end
end

function Distributions.rand(rng::Random.AbstractRNG, d::ZILogPoisson)::Int
  θ, λ = d.θ, d.λ
  c = rand(rng, BernoulliLogit(θ))
  return c == 1 ? 0 : rand(rng, LogPoisson(λ))
end

Distributions.minimum(::ZILogPoisson) = 0
Distributions.maximum(::ZILogPoisson) = Inf
