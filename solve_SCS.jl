using Random
using SparseArrays
using LinearAlgebra
using JuMP
using DataFrames
using CSV
using SCS
using SCS_GPU_jll
using OSQP
function generate_portfolio_example(n::Int, seed::Int=1)
    Random.seed!(seed)
    
    n_assets = n 
    k = Int(n*1)
    F = sprandn(n_assets, k, 1e-4)
    D = spdiagm(0 => rand(n_assets) .* sqrt(k))
    mu = randn(n_assets)
    gamma = 1.0

    optimizer = optimizer_with_attributes(OSQP.Optimizer, "eps_rel" => 1e-6, "eps_abs" => 1, "warm_start" => true, "time_limit_secs" => 3600, "max_iters" => 100000000)
    # Generate JuMP model
    model = Model(optimizer)
    #set_optimizer_attribute(model, "linear_solver", SCS.IndirectSolver)
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
    
        problem = Dict{Symbol, Any}()
        problem[:P] = P
        problem[:q] = q
        problem[:A] = A
        problem[:l] = l
        problem[:u] = u
        problem[:m] = size(A, 1)
        problem[:n] = size(A, 2)
    
        return problem
    end
    
function generate_svm_problem(n::Int, seed::Int=1)
    # 设置随机种子
    Random.seed!(seed)

    # 初始化属性
    n_features = n               # 特征数量
    m_data = Int(n_features*0.5)    # 数据点数量
    N_half = Int(m_data * 0.5)
    gamma_val = 1.0
    b_svm_val = vcat(ones(N_half), -ones(N_half))

    # 生成数据
    A_upp = sprandn(N_half, n_features, 1e-4)
    A_low = sprandn(N_half, n_features, 1e-4)
    A_svm_val = vcat(A_upp / sqrt(n_features) .+ (A_upp .!= 0) / n_features,
                     A_low / sqrt(n_features) .- (A_low .!= 0) / n_features)

    # 生成 QP 问题
    P = spdiagm(0 => vcat(ones(n_features), zeros(m_data)))
    q = vcat(zeros(n_features), (gamma_val) * ones(m_data))

    rowval1 = collect(1:length(b_svm_val))
    colptr1 = collect(1:length(b_svm_val)+1)
    nzval1 = b_svm_val
    rowval2 = collect(1:m_data)
    colptr2 = collect(1:m_data+1)
    nzval2 = ones(m_data)

    A1 = SparseMatrixCSC(length(b_svm_val),length(b_svm_val), colptr1, rowval1, nzval1)
    A2 = SparseMatrixCSC(m_data, m_data, colptr2, rowval2, nzval2)
    A = hcat(-A1 * A_svm_val, A2)

    ru = ones(m_data)
    lb = vcat(-Inf * ones(n_features), zeros(m_data))
    ub = vcat(Inf * ones(n_features), Inf * ones(m_data))
    # Create JuMP model
    optimizer = optimizer_with_attributes(OSQP.Optimizer, "eps_rel" => 1e-6, "eps_abs" => 1, "warm_start" => true, "time_limit_secs" => 3600, "max_iters" => 100000000)
    # Generate JuMP model
    model = Model(optimizer)
    #set_optimizer_attribute(model, "linear_solver", SCS.GpuIndirectSolver)
    # Define variables
    @variable(model, x[1:n + m_data])

    # Define objective
    @objective(model, Min, 0.5 * dot(x, P * x) + dot(q, x))

    # Define constraints

    @constraint(model, A * x .<= ru)

    @constraint(model, lb .<= x .<= ub)
    # Solve the problem
    optimize!(model)

    # Get the solution
    time = MOI.get(model, MOI.SolveTimeSec())
    return time
end
        
function solve_lasso_problem(problem)
    # Extract problem data
    P = problem[:P]
    q = problem[:q]
    A = problem[:A]
    l = problem[:l]
    u = problem[:u]

    # Create JuMP model
    optimizer = optimizer_with_attributes(SCS.Optimizer, "eps_rel" => 1e-6, "eps_abs" => 1, "warm_start" => true, "time_limit_secs" => 3600, "max_iters" => 100000000)
    # Generate JuMP model
    model = Model(optimizer)
    set_optimizer_attribute(model, "linear_solver", SCS.DirectSolver)
    # Define variables
    @variable(model, x[1:problem[:n]])

    # Define objective
    @objective(model, Min, 0.5 * dot(x, P * x) + dot(q, x))

    # Define constraints

    @constraint(model, l .<= A * x .<= u)


    # Solve the problem
    optimize!(model)

    # Get the solution
    time = MOI.get(model, MOI.SolveTimeSec())
    return time
end

function ave_time()
    result_df = DataFrame(scale = Int[], iteration = Int[], time = Float64[])
    for scale in [1000, 10000, 100000, 1000000]
        for i in 1:10
            time = generate_svm_problem(scale, i)
            push!(result_df, (scale, i, time))
            CSV.write("result_Direct.csv", result_df)
        end
    end
    return result_df
end
GPU_id = 2
ENV["CUDA_VISIBLE_DEVICES"] = "$GPU_id"
df = ave_time()
CSV.write("result_SCS_Direct.csv", df)
