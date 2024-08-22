using COPT
using Random
using SparseArrays
using LinearAlgebra
using JuMP
function generate_portfolio_example(n::Int, seed::Int=1)
    Random.seed!(seed)
    
    n_assets = n 
    k = Int(n*1)
    F = sprandn(n_assets, k, 1e-4)
    D = spdiagm(0 => rand(n_assets) .* sqrt(k))
    mu = randn(n_assets)
    gamma = 1.0

    # Generate QP problem
    P = sparse(blockdiag(D, sparse(I, k, k)))
    q = vcat(-mu ./ gamma, zeros(k))
    A = vcat(
        hcat(sparse(ones(1, n_assets)), spzeros(1, k)),
        hcat(F', -sparse(I, k, k)),
    )
    rl = vcat(1.0, zeros(k))

    lb = vcat(zeros(n_assets), -Inf * ones(k))
    ub = vcat(ones(n_assets), Inf * ones(k))

    optimizer = optimizer_with_attributes(COPT.Optimizer, "TimeLimit" => 7200)
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

end

function generate_lasso_problem(n::Int, seed::Int=1)
        # Set random seed
        Random.seed!(seed)
    
        # Initialize parameters
        m = Int(n / 2)
        Ad = sprandn(m, n, 1e-4)
        x_true = (rand(n) .> 0.5) .* randn(n) ./ sqrt(n)
        bd = Ad * x_true + randn(m)
        lambda_max = norm(Ad' * bd, Inf)
        lambda_param = (1/5) * lambda_max
    
        # Construct the QP problem
        P = blockdiag(spzeros(n, n), sparse(2 * I(m)), spzeros(n, n))
        q = vcat(zeros(m + n), lambda_param * ones(n))
        In = I(n)
        Onm = spzeros(n, m)
        A = vcat(hcat(Ad, -I(m), spzeros(m, n)),
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
    
    
function solve_lasso_problem(problem)
    # Extract problem data
    P = problem[:P]
    q = problem[:q]
    A = problem[:A]
    l = problem[:l]
    u = problem[:u]

    # Create JuMP model
    optimizer = optimizer_with_attributes(COPT.Optimizer, "TimeLimit" => 3600)
    # Generate JuMP model
    model = Model(optimizer)

    # Define variables
    @variable(model, x[1:problem[:n]])

    # Define objective
    objective = 0.5 * dot(x, P * x) + dot(q, x)
    @objective(model, Min, objective)

    # Define constraints
    for i in 1:problem[:m]
        row = A[i, :]
        @constraint(model, l[i] <= dot(row, x) <= u[i])
    end

    # Solve the problem
    optimize!(model)

    # Get the solution
    time = MOI.get(model, MOI.SolveTimeSec())
    return time
end

function ave_time()
    mean_time = 0
    for i in 1:10
        problem = generate_lasso_problem(100000, i)
        mean_time += solve_lasso_problem(problem)
    end
    mean_time /= 10
    return mean_time
end 
println(ave_time())
#generate_portfolio_example(100000, 8)