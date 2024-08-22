using Gurobi
using Random
using SparseArrays
using LinearAlgebra
using JuMP
using DataFrames
using CSV
using COPT
function generate_portfolio_example(n::Int, seed::Int=1)
    Random.seed!(seed)
    
    n_assets = n 
    k = Int(n*1)
    F = sprandn(n_assets, k, 1e-4)
    D = spdiagm(0 => rand(n_assets) .* sqrt(k))
    mu = randn(n_assets)
    gamma = 1.0

    optimizer = optimizer_with_attributes(Gurobi.Optimizer, "TimeLimit" => 10000)
    # Generate JuMP model
    model = Model(optimizer)

    @variable(model, x[1:n_assets])
    @variable(model, y[1:k])
    
    # Define the objective function correctly
    @objective(model, Min, sum(D[i,i]*x[i]^2 for i in 1:n_assets) + 
                                     sum(y[j]^2 for j in 1:k) - 
                                     (1 / gamma) * dot(mu, x))
    @constraint(model, sum(x) == 1)
    @constraint(model, F' * x .== y)
    @constraint(model, 0 .<= x .<= 1)

    optimize!(model)
    time = MOI.get(model, MOI.SolveTimeSec())
    return time
end

function generate_lasso_problem(n::Int, seed::Int=1)
    # Set random seed
    # Initialize parameters
    m = Int(n / 2)
    Random.seed!(seed)
    rowval_m = collect(1:m)
    colptr_m = collect(1:m+1)
    nzval_m = ones(m)

    Ad = sprandn(m, n, 1e-4)
    x_true = (rand(n) .> 0.5) .* randn(n) ./ sqrt(n)
    bd = Ad * x_true + randn(m)
    lambda_max = norm(Ad' * bd, Inf)
    lambda_param = (1/5) * lambda_max
    rowval_n = collect(1:n)
    colptr_n = collect(1:n+1)
    nzval_n = ones(n)
    # Construct the QP problem
    P = blockdiag(spzeros(n, n), SparseMatrixCSC(m, m, colptr_m, rowval_m, nzval_m .* 2), spzeros(n, n))
    q = vcat(zeros(m + n), lambda_param * ones(n))
    In = SparseMatrixCSC(n, n, colptr_n, rowval_n, nzval_n)
    Onm = spzeros(n, m)
    A = vcat(hcat(Ad, -SparseMatrixCSC(m, m, colptr_m, rowval_m, nzval_m), spzeros(m, n)),
            hcat(In, Onm, -In),
            hcat(-In, Onm, -In))
    l = vcat(bd, -Inf * ones(n), -Inf * ones(n))
    u = vcat(bd, zeros(n), zeros(n))
    

    optimizer = optimizer_with_attributes(Gurobi.Optimizer)
    # Generate JuMP model
    model = Model(optimizer)
    # Define variables
    @variable(model, x[1:size(A, 2)])

    # Define objective
    @objective(model, Min, 0.5 * dot(x, P * x) + dot(q, x))

    # Define constraints

    @constraint(model, l .<= A * x .<= u)


    # Solve the problem
    print("begin to optimize")
    optimize!(model)

    # Get the solution
    time = MOI.get(model, MOI.SolveTimeSec())
    return time
end

function generate_randomQP_problem(n::Int, seed::Int=1, condition_number::Float64=1e-2, density::Float64=1e-4)
        Random.seed!(seed)
        m = Int(0.5 * n)
        r = 1000
        # Generate problem data
        P = sprandn(r, n, density)
        rowval = collect(1:n)
        colptr = collect(1:n+1)
        nzval = ones(n)
        PtP = P' * P + condition_number * SparseMatrixCSC(n, n, colptr, rowval, nzval)
        #PtP = P' * P
        q = randn(n)
        A = sprandn(m, n, density)
    
        v = randn(n)   # Fictitious solution
        delta = rand(m)  # To get inequality
        ru = A * v + delta
         
        optimizer = optimizer_with_attributes(COPT.Optimizer, "TimeLimit" => 7200)
        # Generate JuMP model
        model = Model(optimizer)

        @variable(model, x[1:n])
        @objective(model, Min, 0.5 * dot(x, PtP * x) + dot(q, x))
        @constraint(model, A * x .<= ru)
    
        optimize!(model)
        time = MOI.get(model, MOI.SolveTimeSec())
        return time
    end

function ave_time()
    result_df = DataFrame(scale = Float64[], iteration = Int[], time = Float64[])
    for scale in [10000, 20000, 30000, 40000, 50000]
        for i in 1:5
            time = generate_randomQP_problem(scale, i, 1e-2, 1e-4) 
            push!(result_df, (scale, i, time))
            CSV.write("result_Gurobi.csv", result_df)
        end
    end
    return result_df
end

df = ave_time()
CSV.write("result_Gurobi.csv", df)