#!/bin/bash
julia --project=@. $(dirname $0)/../forecast.jl
