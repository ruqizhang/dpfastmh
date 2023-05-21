
using Random
using ArgParse
using Statistics
using LinearAlgebra
using Distributions
using StatsBase
using PyPlot
using Seaborn
using JLD, FileIO

include("../util/AliasSampler.jl")
include("../util/SymKL.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--stepsize"
            help = "stepsize"
            arg_type = Float64
            default = 6e-2
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
        "--lam"
            help = "lambda to control minibatch size"
            arg_type = Float64
            default = 10.
        "--K"
            help = "batch size upper bound"
            arg_type = Float64
            default = 60.
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
    samples, total_bs, avg_accept_prob, symkl, kl_time = run_sampler(args,X)
    println("stepsize: $(args["stepsize"])")
    println("lambda: $(args["lam"])")
    println("N: $(args["N"])")
    println("T: $(args["T"])")
    println("Avg Batch Size: $(total_bs/(args["burnin"]+args["nsamples"]))")
    println("Avg Batch Size ratio: $(100*total_bs/args["N"]/(args["burnin"]+args["nsamples"]))")
    println("Avg Acceptance Prob: $(avg_accept_prob)")
    plot(samples)
end

function plot(x::Array{Float64,2})
    scatter(x=x[1,:], y=x[2,:], s=20)
    xlim([-2, 3])
    ylim([-3, 3])
    xticks(fontsize=17)
    yticks(fontsize=17)
    savefig("mog/figs/mog_dpfastmh.pdf")
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
    sampler = RandomWalk(X, args["stepsize"], args["lam"], args["T"], args["epsilon"], args["delta"], args["K"])
    samples, total_bs, avg_accept_prob, symkl, kl_time = new_poissonmh_train(sampler, X, args["nsamples"], args["stepsize"], args["lam"], args["T"], args["burnin"])
    return samples, total_bs, avg_accept_prob, symkl, kl_time
end

struct RandomWalk
    X::Array{Float64,1}
    stepsize::Float64
    data_size::Int64
    psi::Array{Float64,1}
    Psi::Float64
    gamma_A::AliasSampler
    lam::Float64
    T::Float64
    theta_prime::Array{Float64,1}
    sigma1::Float64
    sigma2::Float64
    sigmax::Float64
    max_psi::Float64
    dp_sigma1::Float64
    dp_sigma2::Float64
    epsilon_prime::Float64
    epsilon::Float64
    delta::Float64
    K::Float64
end

function new_poissonmh_train(sampler::RandomWalk, X::Array{Float64,1}, nsamples::Int64, stepsize::Float64, lam::Float64, T::Float64, burnin::Int64)
    d = 2
    theta = zeros(d)
    succ = 0.
    samples = zeros(d, nsamples)
    total_bs = 0.
    total_runtime = 0.
    k = 1
    iters = nsamples+burnin
    interval = 500
    K = Int(iters/interval)
    symkl = zeros(K)
    kl_time = zeros(K)
    for i = 1:iters
        runtime = @elapsed begin
            (theta, sig, bs) = next(sampler, theta)
            total_bs += bs
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
    return samples, total_bs, avg_accept_prob, symkl, kl_time
end

function RandomWalk(X::Array{Float64,1}, stepsize::Float64, lam::Float64, T::Float64, epsilon::Float64, delta::Float64, K::Float64)
    upper = 3.0
    sigma1 = sqrt(10)
    sigma2 = 1
    sigmax = sqrt(2)
    N = length(X)
    d1 = (2*abs.(X).+3*upper)/sigmax^2
    d2 = (abs.(X).+2*upper)/sigmax^2
    psi = sqrt.(d1.^2 + d2.^2)/T
    Psi = sum(psi)
    gamma = Weights(psi ./ Psi)
    theta_prime = zeros(2)
    gamma_A = AliasSampler(gamma)
    max_psi = maximum(psi)
    # K = Psi/(6*max_psi) #4371
    # println(K)
    # exit()
    epsilon_prime = epsilon*Psi/(6*K*max_psi)
    delta_prime = delta*Psi/(2*K*max_psi)
    dp_sigma1 = sqrt(2*log(1.25/delta_prime))/(epsilon_prime)
    dp_sigma2 = sqrt(2*log(1.25/delta))/epsilon
    return RandomWalk(X, stepsize, N, psi, Psi, gamma_A, lam, T, theta_prime, sigma1, sigma2, sigmax, max_psi, dp_sigma1, dp_sigma2, epsilon_prime,epsilon,delta,K);
end

function next(self::RandomWalk, theta::Array{Float64,1})
    data_size = self.data_size
    sig = 0
    theta_prime = proposal(self, theta)
    diff_norm = dist(theta_prime, theta)
    L = diff_norm * self.Psi
    lam = self.lam
    N = lam + L
    s = rand(Poisson(N))
    bs = 0
    logmh = 0.0
    if (N < self.K)
        for ii = 1:s
            idx = rand(self.gamma_A)
            M_i = diff_norm * self.psi[idx]
            (phi_old, phi_new) = get_phi_i(self, idx, M_i, theta, theta_prime)
            ps = (lam*M_i + L*phi_old) / (lam*M_i + L*M_i)
            if (rand() <= ps)
                bs += 1
                logmh += log(1 + L/(lam*M_i)*phi_new) - log(1 + L/(lam*M_i)*phi_old)
            end
        end
        Delta = 2*log(1+self.Psi*diff_norm/lam)
        if (Delta<self.epsilon_prime)
            acc_prob = stand_mh(self,logmh)
        else
            acc_prob = private_tunamh(self,logmh,Delta)
        end
    else
        s = data_size
        logmh = 0.0
        for i = 1:data_size
            ll_old = logtarget(self, theta, i)
            ll_new = logtarget(self, theta_prime, i)
            logmh += (ll_new - ll_old)
        end
        Delta2 = 2*self.max_psi * diff_norm
        if (Delta2<self.epsilon)
            acc_prob = stand_mh(self,logmh)
        else
            acc_prob = private_mh(self,logmh,Delta2)
        end
    end
    if rand() < acc_prob
        theta .= theta_prime
        sig = 1
    end
    return (theta, sig, s)
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

function private_tunamh(self::RandomWalk, u::Float64,Delta::Float64)
    sd = self.dp_sigma1*Delta
    noise = sd*rand()
    return exp(u + noise - sd^2/2)
end

function private_mh(self::RandomWalk, u::Float64,Delta::Float64)
    sd = self.dp_sigma2*Delta
    noise = sd*rand()
    return exp(u + noise - sd^2/2)
end

function get_phi_i(self::RandomWalk,idx::Int64,Mi::Float64,theta::Array{Float64,1},theta_prime::Array{Float64,1})
    f1 = exp(-(self.X[idx] - theta[1])^2 / self.sigmax^2 / 2)
    f2 = exp(-(self.X[idx] - theta[1] - theta[2])^2 / self.sigmax^2 / 2)
    logl = log(f1 + f2)/self.T

    f1 = exp(-(self.X[idx] - theta_prime[1])^2 / self.sigmax^2 / 2)
    f2 = exp(-(self.X[idx] - theta_prime[1] - theta_prime[2])^2 / self.sigmax^2 / 2)
    logl_prime = log(f1 + f2)/self.T
    return (0.5*(logl - logl_prime + Mi), 0.5*(logl_prime - logl + Mi))
end

function logtarget(self::RandomWalk,theta::Array{Float64,1},idx::Int64)
    f1 = exp(-(self.X[idx] - theta[1])^2 / self.sigmax^2 / 2)
    f2 = exp(-(self.X[idx] - theta[1] - theta[2])^2 / self.sigmax^2 / 2)
    logp = log(f1 + f2)/self.T
    return logp
end

main()