module BaerbelbotPredict

using DotEnv, HTTP, JSON3, DataFrames, Dates, CategoricalArrays, LazyArrays, Distributions, Turing, Random, ReverseDiff, Memoization, MCMCChains, StatsFuns, Compose, Gadfly

export get_data, prepare_data!
export ZILogPoisson, ZIPoisReg, fit, forecast, plot_forecast

include("ZILogPoisson.jl")

function get_data(url::String)::DataFrame
  response = HTTP.get(url)
  response.status == 200 || error("invalid response")
  return response.body |> JSON3.read |> DataFrame
end

function prepare_data!(data::DataFrame)
  data.date = @. Date(data.date, dateformat"y-m-dTH:M:S.sZ")
  data.t = date2index(data.date)
  data.u = user2index(data.userId)
  fill_data!(data)
  fix_negative_amounts!(data)
  cumulate_amounts!(data)
  return nothing
end

function fix_negative_amounts!(data::DataFrame)
  data.amount = [maximum([x, 0]) for x in data.amount]
end

function cumulate_amounts!(data::DataFrame)
  sort!(data, [:userId, :date])
  cumulative_amounts = combine(groupby(data, :userId), :amount => cumsum => :cum_amount)
  data.cum_amount = cumulative_amounts.cum_amount
  return nothing
end

function fill_data!(data::DataFrame)
  for date in unique(data.date)
    date_index = findfirst(x -> x == date, data.date)
    date_row = data[date_index, :]

    for user_id in unique(data.userId)
      user_index = findfirst(x -> x == user_id, data.userId)
      user_row = data[user_index, :]
      match = any(@. (data.date == date) & (data.userId == user_id))

      if (!match)
        insert_row = DataFrame(
          userId = user_id,
          username = user_row.username,
          date = date,
          u = user_row.u,
          t = date_row.t,
          amount = 0
        )
        append!(data, insert_row)
      end
    end
  end
  return nothing
end

function date2index(dates::Vector{Date})::Vector{Int}
  min = minimum(dates)
  weeks = round.(dates .- min, Week(1))
  index = [x.value + 1 for x in weeks]
  index = Int.(categorical(index).refs)  # condense
  return index
end

function user2index(users::Vector{String})::Vector{Int}
  Int.(categorical(users).refs)
end

@model function ZIPoisReg(y::Vector{Int}, t::Vector{Int}, person_ids::Vector{Int}; n = length(y), n_t = length(unique(t)), n_person = length(unique(person_ids)))
  # predictors for poisson expectation
  a_0 ~ Normal(log(2.5), log(2.0))
  σ_a_person ~ truncated(Normal(), 0, Inf)
  a_person ~ filldist(Normal(0, σ_a_person), n_person)
  σ_t ~ truncated(Normal(), 0, Inf)
  a_t ~ filldist(Normal(0, σ_t), n_t)

  # predictors for mixing probability
  b_0 ~ Normal(logit(0.1), 1)
  σ_b_person ~ truncated(Normal(), 0, Inf)
  b_person ~ filldist(Normal(0, σ_b_person), n_person)

  y ~ arraydist(LazyArray(@~ @. ZILogPoisson(
    b_0 + b_person[person_ids],
    a_0 + a_person[person_ids] + a_t[t]
  )))
end

function fit(data::DataFrame; algorithm = NUTS(0.65), I = 2000)
  model = ZIPoisReg(data.amount, data.t, data.u)
  chain = sample(model, algorithm, I)
  return chain
end

function forecast(chain::Chains, data::DataFrame; to = endofyear())
  forecasts = []

  for userId in unique(data.userId)
    from = maximum(data[data.userId.==userId, :date])
    forecast_range = from:Week(1):to
    forecast_length = length(forecast_range)
    Base.push!(forecasts, forecast(chain, data, userId, forecast_length))
  end

  forecast_df = vcat(forecasts...)
  filter!(x -> x.date <= to, forecast_df)
end

function forecast(chain::Chains, data::DataFrame, userId::String, nsteps)
  person_data = filter(x -> x.userId == userId, data)
  person_id = first(unique(person_data.u))

  predictions = forecast_nsteps(chain, person_id, nsteps)
  max = fill(maximum(person_data.cum_amount), 1, 2000)  # add this for offset and plotting
  predictions = vcat(max, predictions)
  cum_predictions = cumsum(predictions, dims = 1)

  result = DataFrame(
    u = person_id,
    username = first(unique(person_data.username)),
    userId = userId,
    lwr = [quantile(cum_predictions[i, :], 0.1) for i in 1:nsteps+1],
    md = [quantile(cum_predictions[i, :], 0.5) for i in 1:nsteps+1],
    upr = [quantile(cum_predictions[i, :], 0.9) for i in 1:nsteps+1],
    t = maximum(person_data.t):maximum(person_data.t)+nsteps,
    date = maximum(person_data.date):Week(1):maximum(person_data.date)+Week(nsteps)
  )

  return result
end

function forecast_nsteps(chain, person_id, n_steps)
  predictions = Matrix{Int}(undef, n_steps, 2000)
  for i in 1:n_steps
    predictions[i, :] = forecast_1step(chain, person_id)
  end
  return predictions
end

function forecast_1step(chain, person_id)
  df = DataFrame(chain)
  a_t = @. rand(Normal(0, df[:, :σ_t]))
  y_pred = @. rand(ZILogPoisson(
    df[:, :b_0] + df[:, Symbol("b_person[$person_id]")],
    df[:, :a_0] + df[:, Symbol("a_person[$person_id]")] + a_t)
  )
  return y_pred
end

function plot_forecast(chain::Chains, data::DataFrame)
  p = plot(data, x = :date, y = :cum_amount, color = :username, Geom.line,
    Guide.xlabel("Datum"),
    Guide.ylabel("Anzahl Bier"),
    Guide.colorkey(title = "")
  )
  Gadfly.push!(p, layer(y = :md, ymin = :lwr, ymax = :upr, color = :username, alpha = [0.5], Geom.path, Geom.ribbon))

  # for single persons add statistics
  if length(unique(data.u)) == 1
    person_id = first(data.u)
    predictions = data[ismissing.(data.amount), :]
    upcoming = round(Int, predictions[2, :md] - predictions[1, :md])
    p_attend = round((1 - missing_probability(chain, person_id)), digits = 2)
    Gadfly.push!(p,
      Guide.annotation(compose(Compose.context(),
        Compose.text(0.5cm, 0.8cm, "E(Bier) = $(upcoming)"),
        fill(Gadfly.RGB(0.627, 0.627, 0.627)), Compose.stroke(nothing), fontsize(9pt)
      )),
      Guide.annotation(compose(Compose.context(),
        Compose.text(0.5cm, 1.3cm, "P(anwesend) = $(p_attend)"),
        fill(Gadfly.RGB(0.627, 0.627, 0.627)), Compose.stroke(nothing), fontsize(9pt)
      ))
    )
  end

  return p
end

function missing_probability(chain::Chains, person_id::Int)
  df = DataFrame(chain)
  b_0 = df.b_0
  b_person = df[:, Symbol("b_person[$person_id]")]
  prob = @. logistic(b_0 + b_person)
  return mean(prob)
end


function endofyear()
  return Date(year(today()), 12, 31)
end

# function save_forecast!(chain, df)
#   # save chains
#   h5open("data/mcmc-chains.h5", "w") do f
#     write(f, chain)
#   end
#   # save forecasted data
#   open("data/data.csv", "w") do f
#     CSV.write(f, df)
#   end
#   return nothing
# end

# function read_forecast(; chain = "data/mcmc-chains.h5", df = "data.csv")
#   saved_chain = h5open(chain, "r") do f
#     read(f, Chains)
#   end
#   saved_df = open(df, "r") do f
#     tmp = CSV.read(f, DataFrame)
#     tmp.uid = string.(tmp.uid)
#   end
#   return saved_chain, saved_df
# end


end # module
