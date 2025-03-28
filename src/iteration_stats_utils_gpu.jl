
mutable struct CuDualStats
    dual_objective::Float64
    dual_residual::CuVector{Float64}
    reduced_costs::CuVector{Float64}
end

mutable struct CuBufferKKTState
    primal_solution::CuVector{Float64}
    dual_solution::CuVector{Float64}
    primal_product::CuVector{Float64}
    primal_gradient::CuVector{Float64}
    primal_obj_product::CuVector{Float64} 
    lower_variable_violation::CuVector{Float64}
    upper_variable_violation::CuVector{Float64}
    constraint_violation::CuVector{Float64}
    dual_objective_contribution_array::CuVector{Float64}
    reduced_costs_violation::CuVector{Float64}
    dual_stats::CuDualStats
    dual_res_inf::Float64
end

function compute_primal_residual_constraint_kernel!(
    activities::CuDeviceVector{Float64},
    right_hand_side::CuDeviceVector{Float64},
    num_equalities::Int64,
    num_constraints::Int64,
    constraint_violation::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_equalities
        @inbounds begin
            constraint_violation[tx] = right_hand_side[tx] - activities[tx]
        end
    end
    if (num_equalities + 1) <= tx <= num_constraints
        @inbounds begin
            constraint_violation[tx] = max(right_hand_side[tx] - activities[tx], 0.0)
        end
    end

    return 
end


function compute_primal_residual_variable_kernel!(
    primal_vec::CuDeviceVector{Float64},
    variable_lower_bound::CuDeviceVector{Float64},
    variable_upper_bound::CuDeviceVector{Float64},
    num_variables::Int64,
    lower_variable_violation::CuDeviceVector{Float64},
    upper_variable_violation::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_variables
        @inbounds begin
            lower_variable_violation[tx] = max(variable_lower_bound[tx] - primal_vec[tx], 0.0)
            upper_variable_violation[tx] = max(primal_vec[tx] - variable_upper_bound[tx], 0.0)
        end
    end

    return 
end


function compute_primal_residual!(
    problem::CuQuadraticProgrammingProblem,
    buffer_kkt::CuBufferKKTState,
)
    NumBlockPrimal = ceil(Int64, problem.num_variables/ThreadPerBlock)
    NumBlockDual = ceil(Int64, problem.num_constraints/ThreadPerBlock)

    CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockPrimal compute_primal_residual_variable_kernel!(
        buffer_kkt.primal_solution,
        problem.variable_lower_bound,
        problem.variable_upper_bound,
        problem.num_variables,
        buffer_kkt.lower_variable_violation,
        buffer_kkt.upper_variable_violation,
    )

    CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockDual compute_primal_residual_constraint_kernel!(
        buffer_kkt.primal_product,
        problem.right_hand_side,
        problem.num_equalities,
        problem.num_constraints,
        buffer_kkt.constraint_violation,
    )
end
      

function primal_obj(
    problem::CuQuadraticProgrammingProblem,
    primal_solution::CuVector{Float64},
    primal_obj_product::CuVector{Float64},
)
    return problem.objective_constant +
        CUDA.dot(problem.objective_vector, primal_solution) +
        0.5 * CUDA.dot(primal_solution, primal_obj_product)
end


function reduced_costs_dual_objective_contribution_kernel!(
    variable_lower_bound::CuDeviceVector{Float64},
    variable_upper_bound::CuDeviceVector{Float64},
    reduced_costs::CuDeviceVector{Float64},
    num_variables::Int64,
    dual_objective_contribution_array::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_variables
        @inbounds begin
            if reduced_costs[tx] > 0.0
                dual_objective_contribution_array[tx] = variable_lower_bound[tx] * reduced_costs[tx]
            elseif reduced_costs[tx] < 0.0
                dual_objective_contribution_array[tx] = variable_upper_bound[tx] * reduced_costs[tx]
            else
                dual_objective_contribution_array[tx] = 0.0
            end
        end
    end

    return 
end


function reduced_costs_dual_objective_contribution(
    problem::CuQuadraticProgrammingProblem,
    buffer_kkt::CuBufferKKTState,
)
    NumBlockPrimal = ceil(Int64, problem.num_variables/ThreadPerBlock)

    CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockPrimal reduced_costs_dual_objective_contribution_kernel!(
        problem.variable_lower_bound,
        problem.variable_upper_bound,
        buffer_kkt.dual_stats.reduced_costs,
        problem.num_variables,
        buffer_kkt.dual_objective_contribution_array,
    )  
 
    dual_objective_contribution = sum(buffer_kkt.dual_objective_contribution_array)

    return dual_objective_contribution
end


function compute_reduced_costs_from_primal_gradient_kernel!(
    primal_gradient::CuDeviceVector{Float64},
    isfinite_variable_lower_bound::CuDeviceVector{Bool},
    isfinite_variable_upper_bound::CuDeviceVector{Bool},
    num_variables::Int64,
    reduced_costs::CuDeviceVector{Float64},
    reduced_costs_violation::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_variables
        @inbounds begin
            reduced_costs[tx] = max(primal_gradient[tx], 0.0) * isfinite_variable_lower_bound[tx] + min(primal_gradient[tx], 0.0) * isfinite_variable_upper_bound[tx]

            reduced_costs_violation[tx] = primal_gradient[tx] - reduced_costs[tx]
        end
    end

    return 
end

"""
Compute reduced costs from primal gradient
"""
function compute_reduced_costs_from_primal_gradient!(
    problem::CuQuadraticProgrammingProblem,
    buffer_kkt::CuBufferKKTState,
)
    NumBlockPrimal = ceil(Int64, problem.num_variables/ThreadPerBlock)
    CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockPrimal compute_reduced_costs_from_primal_gradient_kernel!(
        buffer_kkt.primal_gradient,
        problem.isfinite_variable_lower_bound,
        problem.isfinite_variable_upper_bound,
        problem.num_variables,
        buffer_kkt.dual_stats.reduced_costs,
        buffer_kkt.reduced_costs_violation,
    )  
end


function compute_dual_residual_kernel!(
    dual_solution::CuDeviceVector{Float64},
    num_equalities::Int64,
    num_inequalities::Int64,
    dual_residual::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_inequalities
        @inbounds begin
            dual_residual[tx] = max(-dual_solution[tx+num_equalities], 0.0)
        end
    end
    return 
end


function compute_dual_stats!(
    problem::CuQuadraticProgrammingProblem,
    buffer_kkt::CuBufferKKTState,
)
    compute_reduced_costs_from_primal_gradient!(problem, buffer_kkt)

    NumBlockIneq = ceil(Int64, (problem.num_constraints-problem.num_equalities)/ThreadPerBlock)

    if NumBlockIneq >= 1 
        CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockIneq compute_dual_residual_kernel!(
            buffer_kkt.dual_solution,
            problem.num_equalities,
            problem.num_constraints - problem.num_equalities,
            buffer_kkt.dual_stats.dual_residual,
        )  
    end

    buffer_kkt.dual_res_inf = CUDA.norm([buffer_kkt.dual_stats.dual_residual; buffer_kkt.reduced_costs_violation], Inf)

    base_dual_objective = CUDA.dot(problem.right_hand_side, buffer_kkt.dual_solution) + problem.objective_constant - 0.5 * CUDA.dot(buffer_kkt.primal_solution, buffer_kkt.primal_obj_product)

    buffer_kkt.dual_stats.dual_objective = base_dual_objective + reduced_costs_dual_objective_contribution(problem, buffer_kkt)
end

function corrected_dual_obj(buffer_kkt::CuBufferKKTState)
    if buffer_kkt.dual_res_inf == 0.0
        return buffer_kkt.dual_stats.dual_objective
    else
        return -Inf
    end
end

"""
Compute convergence information of the given primal and dual solutions
"""
function compute_convergence_information(
    problem::CuQuadraticProgrammingProblem,
    qp_cache::CachedQuadraticProgramInfo,
    primal_iterate::CuVector{Float64},
    dual_iterate::CuVector{Float64},
    eps_ratio::Float64,
    candidate_type::PointType,
    primal_product::CuVector{Float64},
    dual_product::CuVector{Float64}, 
    primal_gradient::CuVector{Float64},
    primal_obj_product::CuVector{Float64},
    buffer_kkt::CuBufferKKTState,
)
    
    ## construct buffer_kkt
   
    CUDA.copyto!(buffer_kkt.primal_solution, primal_iterate)
    CUDA.copyto!(buffer_kkt.dual_solution, dual_iterate)
    CUDA.copyto!(buffer_kkt.primal_product, primal_product)
    CUDA.copyto!(buffer_kkt.primal_gradient, primal_gradient)
    CUDA.copyto!(buffer_kkt.primal_obj_product, primal_obj_product)

    convergence_info = ConvergenceInformation()

    compute_primal_residual!(problem, buffer_kkt)
    convergence_info.primal_objective = primal_obj(problem, buffer_kkt.primal_solution, buffer_kkt.primal_obj_product)
    primal_residual_vector = [buffer_kkt.constraint_violation; buffer_kkt.lower_variable_violation; buffer_kkt.upper_variable_violation]
    convergence_info.l_inf_primal_residual = CUDA.norm(primal_residual_vector, Inf)
    convergence_info.l2_primal_residual = sqrt(CUDA.dot(primal_residual_vector,primal_residual_vector))
    convergence_info.relative_l_inf_primal_residual =
        convergence_info.l_inf_primal_residual /
        (eps_ratio + max(qp_cache.l_inf_norm_primal_right_hand_side, CUDA.norm(buffer_kkt.primal_product, Inf))) 
    convergence_info.relative_l2_primal_residual =
        convergence_info.l2_primal_residual /
        (eps_ratio + qp_cache.l2_norm_primal_right_hand_side + sqrt(CUDA.dot(buffer_kkt.primal_product,buffer_kkt.primal_product)))
    convergence_info.l_inf_primal_variable = CUDA.norm(buffer_kkt.primal_solution, Inf)
    convergence_info.l2_primal_variable = sqrt(CUDA.dot(buffer_kkt.primal_solution,buffer_kkt.primal_solution))

    

    compute_dual_stats!(problem, buffer_kkt)
    convergence_info.dual_objective = buffer_kkt.dual_stats.dual_objective
    convergence_info.l_inf_dual_residual = buffer_kkt.dual_res_inf
    convergence_info.l2_dual_residual = sqrt(CUDA.dot([buffer_kkt.dual_stats.dual_residual; buffer_kkt.reduced_costs_violation],[buffer_kkt.dual_stats.dual_residual; buffer_kkt.reduced_costs_violation]))
    convergence_info.relative_l_inf_dual_residual =
        convergence_info.l_inf_dual_residual /
        (eps_ratio + max(qp_cache.l_inf_norm_primal_linear_objective, CUDA.norm(buffer_kkt.primal_obj_product, Inf), CUDA.norm(dual_product, Inf)))
    convergence_info.relative_l2_dual_residual =
        convergence_info.l2_dual_residual /
        (eps_ratio + qp_cache.l2_norm_primal_linear_objective + sqrt(CUDA.dot(buffer_kkt.primal_obj_product,buffer_kkt.primal_obj_product)) + sqrt(CUDA.dot(dual_product,dual_product)))
    convergence_info.l_inf_dual_variable = CUDA.norm(buffer_kkt.dual_solution, Inf)
    convergence_info.l2_dual_variable = sqrt(CUDA.dot(buffer_kkt.dual_solution,buffer_kkt.dual_solution))

    convergence_info.corrected_dual_objective = corrected_dual_obj(buffer_kkt)

    gap = abs(convergence_info.primal_objective - convergence_info.dual_objective)
    abs_obj =
        max(abs(convergence_info.primal_objective) ,
        abs(convergence_info.dual_objective))
    
    convergence_info.relative_optimality_gap = gap / (eps_ratio + abs_obj)

    convergence_info.candidate_type = candidate_type

    return convergence_info
end


"""
Compute iteration stats of the given primal and dual solutions
"""
function compute_iteration_stats(
    problem::CuQuadraticProgrammingProblem,
    qp_cache::CachedQuadraticProgramInfo,
    primal_iterate::CuVector{Float64},
    dual_iterate::CuVector{Float64},
    iteration_number::Integer,
    cumulative_kkt_matrix_passes::Float64,
    cumulative_time_sec::Float64,
    eps_optimal_absolute::Float64,
    eps_optimal_relative::Float64,
    step_size::Float64,
    primal_weight::Float64,
    candidate_type::PointType,
    primal_product::CuVector{Float64},
    dual_product::CuVector{Float64}, 
    primal_gradient::CuVector{Float64},
    primal_obj_product::CuVector{Float64},
    buffer_kkt::CuBufferKKTState,
)
    stats = IterationStats()
    stats.iteration_number = iteration_number
    stats.cumulative_kkt_matrix_passes = cumulative_kkt_matrix_passes
    stats.cumulative_time_sec = cumulative_time_sec

    stats.convergence_information = [
        compute_convergence_information(
            problem,
            qp_cache,
            primal_iterate,
            dual_iterate,
            eps_optimal_absolute / eps_optimal_relative,
            candidate_type,
            primal_product,
            dual_product, 
            primal_gradient,
            primal_obj_product,
            buffer_kkt,
        ),
    ]
    
    stats.step_size = step_size
    stats.primal_weight = primal_weight
    stats.method_specific_stats = Dict{AbstractString,Float64}()

    return stats
end

mutable struct CuBufferOriginalSol
    original_primal_solution::CuVector{Float64}
    original_dual_solution::CuVector{Float64}
    original_primal_product::CuVector{Float64}
    original_dual_product::CuVector{Float64} #
    original_primal_gradient::CuVector{Float64}
    original_primal_obj_product::CuVector{Float64}
end

"""
Compute the iteration stats of the unscaled primal and dual solutions
"""
function evaluate_unscaled_iteration_stats(
    scaled_problem::CuScaledQpProblem,
    qp_cache::CachedQuadraticProgramInfo,
    termination_criteria::TerminationCriteria,
    record_iteration_stats::Bool,
    primal_solution::CuVector{Float64},
    dual_solution::CuVector{Float64},
    iteration::Int64,
    cumulative_time::Float64,
    cumulative_kkt_passes::Float64,
    eps_optimal_absolute::Float64,
    eps_optimal_relative::Float64,
    step_size::Float64,
    primal_weight::Float64,
    candidate_type::PointType,
    primal_product::CuVector{Float64},
    dual_product::CuVector{Float64}, 
    primal_gradient::CuVector{Float64},
    primal_obj_product::CuVector{Float64}, 
    buffer_original::CuBufferOriginalSol,
    buffer_kkt::CuBufferKKTState,
)
    # Unscale iterates. 
    buffer_original.original_primal_solution .=
        primal_solution ./ scaled_problem.variable_rescaling
    buffer_original.original_primal_solution .*= scaled_problem.constant_rescaling

    buffer_original.original_primal_gradient .=
        primal_gradient .* scaled_problem.variable_rescaling
    buffer_original.original_primal_gradient .*= scaled_problem.constant_rescaling

    buffer_original.original_dual_solution .=
        dual_solution ./ scaled_problem.constraint_rescaling
    buffer_original.original_dual_solution .*= scaled_problem.constant_rescaling

    buffer_original.original_primal_product .=
        primal_product .* scaled_problem.constraint_rescaling
    buffer_original.original_primal_product .*= scaled_problem.constant_rescaling

    buffer_original.original_dual_product .=
        dual_product .* scaled_problem.variable_rescaling
    buffer_original.original_dual_product .*= scaled_problem.constant_rescaling

    buffer_original.original_primal_obj_product .=
        primal_obj_product .* scaled_problem.variable_rescaling
    buffer_original.original_primal_obj_product .*= scaled_problem.constant_rescaling

    return compute_iteration_stats(
        scaled_problem.original_qp,
        qp_cache,
        buffer_original.original_primal_solution,
        buffer_original.original_dual_solution,
        iteration - 1,
        cumulative_kkt_passes,
        cumulative_time,
        eps_optimal_absolute,
        eps_optimal_relative,
        step_size,
        primal_weight,
        candidate_type,
        buffer_original.original_primal_product,
        buffer_original.original_dual_product,
        buffer_original.original_primal_gradient,
        buffer_original.original_primal_obj_product,
        buffer_kkt,
    )
end