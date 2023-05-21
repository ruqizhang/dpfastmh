
using Random
using ArgParse
using Statistics
using LinearAlgebra
using Distributions
using StatsBase
using PyPlot
using Seaborn
using JLD, FileIO

include("../util/SymKL.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--stepsize"
            help = "stepsize"
            arg_type = Float64
            default = 8e-2 
        "--N"
            help = "dataset size"
            arg_type = Int64
            default = 50000
        "--T"
            help = "temperature"
            arg_type = Float64
            default = 500.0
        "--nsamples"
            help = "number of samples"
            arg_type = Int64
            default = 20000
        "--burnin"
            help = "number of samples as burnin"
            arg_type = Int64
            default = 0
        "--epsilon"
            help = "privacy parameter"
            arg_type = Float64
            default = 0.05
        "--delta"
            help = "privacy parameter"
            arg_type = Float64
            default = 1e-5
    end

    return parse_args(s)
end

function main() 
    args = parse_commandline()
    X = generate_data(args)
    samples, avg_accept_prob, symkl, kl_time = run_sampler(args,X)
    println("Avg Acceptance Prob: $(avg_accept_prob)")
    plot(samples)
end

function plot(x::Array{Float64,2})
    scatter(x=x[1,:], y=x[2,:], s=20)
    xlim([-2, 3])
    ylim([-3, 3])
    xticks(fontsize=17)
    yticks(fontsize=17)
    savefig("mog/figs/mog_dpfastmh_full.pdf")
end

function generate_data(args::Dict)
    N = args["N"]
    Random.seed!(1111)
    sigmax = sqrt(2)
    X = zeros(N)
    for i = 1:N
        if rand()<0.5
            X[i] = randn()*sigmax
        else
            X[i] = randn()*sigmax + 1
        end
    end
    return X
end

function run_sampler(args::Dict, X::Array{Float64,1})
    sampler = RandomWalk(X, args["stepsize"], args["T"], args["epsilon"], args["delta"])
    samples, avg_accept_prob, symkl, kl_time = mh_train(sampler, X, args["nsamples"], args["stepsize"], args["T"], args["burnin"])
    return samples, avg_accept_prob, symkl, kl_time
end

struct RandomWalk
    X::Array{Float64,1}
    stepsize::Float64
    data_size::Int64
    T::Float64
    theta_prime::Array{Float64,1}
    sigma1::Float64
    sigma2::Float64
    sigmax::Float64
    max_psi::Float64
    dp_sigma::Float64
    epsilon::Float64
end

function mh_train(sampler::RandomWalk, X::Array{Float64,1}, nsamples::Int64, stepsize::Float64, T::Float64, burnin::Int64)
    d = 2
    theta = zeros(d)
    succ = 0.
    samples = zeros(d, nsamples)
    total_runtime = 0.
    k = 1
    iters = nsamples+burnin
    interval = 500
    K = Int(iters/interval)
    symkl = zeros(K)
    kl_time = zeros(K)
    for i = 1:iters
        runtime = @elapsed begin
            (theta, sig) = next(sampler, theta)
            succ += sig
            if i > burnin
                for j = 1:d
                    samples[j,i-burnin] = theta[j]
                end
            end
        end
        total_runtime += runtime
        if (i % interval == 1)
            symkl[k] = sym_KL(samples[:,1:i-burnin])[1]
            println(symkl[k])
            kl_time[k] = total_runtime
            k += 1
        end
    end
    avg_accept_prob = float(succ) / iters
    return samples, avg_accept_prob, symkl, kl_time
end

function RandomWalk(X::Array{Float64,1}, stepsize::Float64, T::Float64,epsilon::Float64, delta::Float64)
    upper = 3.0
    sigma1 = sqrt(10)
    sigma2 = 1
    sigmax = sqrt(2)
    N = length(X)
    theta_prime = zeros(2)
    d1 = (2*abs.(X).+3*upper)/sigmax^2
    d2 = (abs.(X).+2*upper)/sigmax^2
    psi = sqrt.(d1.^2 + d2.^2)/T
    max_psi = maximum(psi)
    dp_sigma = sqrt(2*log(1.25/delta))/epsilon
    return RandomWalk(X, stepsize, length(X), T, theta_prime, sigma1, sigma2, sigmax,max_psi,dp_sigma,epsilon);
end

function next(self::RandomWalk, theta::Array{Float64,1})
    data_size = self.data_size
    sig = 0
    theta_prime = proposal(self, theta)
    diff_norm = dist(theta_prime, theta)
    logmh = 0.0
    for i = 1:data_size
        ll_old = logtarget(self, theta, i)
        ll_new = logtarget(self, theta_prime, i)
        logmh += (ll_new - ll_old)
    end
    Delta = 2*self.max_psi * diff_norm
    if (Delta<self.epsilon)
        acc_prob = stand_mh(self,logmh)
    else
        acc_prob = private_mh(self,logmh,Delta)
    end
    if rand() < acc_prob
        theta .= theta_prime
        sig = 1
    end
    return (theta, sig)
end

function dist(x::Array{Float64,1}, y::Array{Float64,1})
    @assert(length(x) == length(y))
    acc = 0;
    for i = 1:length(x)
        acc += (x[i] - y[i])^2
    end
    return sqrt(acc)
end

function proposal(self::RandomWalk, theta::Array{Float64,1})
    for i = 1:length(theta)
        self.theta_prime[i] = theta[i] + self.stepsize * randn()
    end
    return self.theta_prime
end

function stand_mh(self::RandomWalk, u::Float64)
    return exp(u)
end

function private_mh(self::RandomWalk, u::Float64,Delta::Float64)
    sd = self.dp_sigma*Delta
    noise = sd*rand()
    return exp(u + noise - sd^2/2)
end

function logtarget(self::RandomWalk,theta::Array{Float64,1},idx::Int64)
    f1 = exp(-(self.X[idx] - theta[1])^2 / self.sigmax^2 / 2)
    f2 = exp(-(self.X[idx] - theta[1] - theta[2])^2 / self.sigmax^2 / 2)
    logp = log(f1 + f2)/self.T
    return logp
end

function logprior(self::RandomWalk,theta::Array{Float64,1},theta_prime::Array{Float64,1})
    prior0 = exp(-theta[1]^2 / self.sigma1^2 / 2 -theta[2]^2 / self.sigma2^2 / 2)
    prior0 *= 1 / sqrt(2 * pi) / (self.sigma1*self.sigma2)
    logprior0 = log(prior0)

    prior = exp(-theta_prime[1]^2 / self.sigma1^2 / 2 -theta_prime[2]^2 / self.sigma2^2 / 2)
    prior *= 1 / sqrt(2 * pi) / (self.sigma1*self.sigma2)
    logprior = log(prior)
    return logprior - logprior0
end

main()