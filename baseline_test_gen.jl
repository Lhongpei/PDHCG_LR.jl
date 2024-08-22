using JuMP
using SCS
using QPSReader
using CSV
using DataFrames
using SCS_GPU_jll
using JuMP

function solve_gen_problem(problem, accuracy::Float64 = 1e-6, gpu_on ::Int = 0)
    # Extract problem data
    P = problem[:P]
    q = problem[:q]
    A = problem[:A]
    l = problem[:l]
    u = problem[:u]
    print('1')
    optimizer = optimizer_with_attributes(SCS.Optimizer, "eps_rel" => accuracy, "eps_abs" => 1, "warm_start" => false, "time_limit_secs" => 3600, "max_iters" => 100000000)
    model = Model(optimizer)
    if gpu_on == 1
        set_optimizer_attribute(model, "linear_solver", SCS.GpuIndirectSolver)
    else
        set_optimizer_attribute(model, "linear_solver", SCS.IndirectSolver)
    end
    # Create JuMP model
    #model = Model(Ipopt.Optimizer)
    
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

    # # Get the solution
    # solution = value.(x)
    # return solution
end

include("problem_gen_with_juMP.jl")
# Example usage
save_file = "results_gen_1e-6.csv"
acc = 1e-10
GPU_on = 1
GPU_id = 1
ENV["CUDA_VISIBLE_DEVICES"] = "$GPU_id"
problem = generate_problem_data_randomQP(10000, 42)
solve_gen_problem(problem, acc, GPU_on)
#main(qps_folder, save_file, acc, GPU_on)

