using JLD, FileIO

function sym_KL(x::Array{Float64,2})
    samples = load("mog/checkpoints/samples_2e5.jld")["data"]
    xedges = range(minimum(samples[1,:]), maximum(samples[1,:]), length=100)
    yedges = range(minimum(samples[2,:]), maximum(samples[2,:]), length=100)
    h1 = fit(Histogram, (x[1,:], x[2,:]), (xedges, yedges)).weights / size(x, 2)
    h2 = fit(Histogram, (samples[1,:], samples[2,:]), (xedges, yedges)).weights / size(samples, 2)
    EPSILON = 10e-14
    h1_log = log.(clamp.(h1, EPSILON, 100))
    h2_log = log.(clamp.(h2, EPSILON, 100))
    h1_h2 = sum(sum(h1 .* (h1_log - h2_log), dims=1), dims=2)
    h2_h1 = sum(sum(h2 .* (h2_log - h1_log), dims=1), dims=2)
    return (h1_h2+h2_h1)/2
end