using BaerbelbotPredict, DotEnv, Turing, ReverseDiff, Memoization, Gadfly, Cairo, Fontconfig

Gadfly.push_theme(:dark)

Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

conf = DotEnv.config()

hash_file = joinpath(conf["SHARED_DATA_DIR"], "data_hash")
plot_path = joinpath(conf["SHARED_DATA_DIR"], "plots")

data = get_data(conf["API_URL"])
data_hash = hash(data)

previous_hash = isfile(hash_file) ? reinterpret(UInt64, read(hash_file)) |> first : ""

if data_hash != previous_hash
  prepare_data!(data)

  chain = BaerbelbotPredict.fit(data)

  prediction = forecast(chain, data)
  prediction_data = vcat(data, prediction, cols = :union)

  # plot predictions
  for user in unique(data.userId)
    plot_data = filter(x -> x.userId == user, prediction_data)
    p = plot_forecast(chain, plot_data)
    p |> PNG(joinpath(plot_path, "$(user).png"), 15cm, 10cm, dpi = 250)
  end

  write(hash_file, data_hash)
else
  @info "Skipped forecasting because data has not changed..."
end
