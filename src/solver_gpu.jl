mutable struct CuPdhcgSolverState
    current_primal_solution::CuVector{Float64}
    current_dual_solution::CuVector{Float64}
    delta_primal::CuVector{Float64}
    delta_dual::CuVector{Float64}
    current_primal_product::CuVector{Float64}
    current_dual_product::CuVector{Float64}
    current_primal_obj_product::CuVector{Float64} 
    solution_weighted_avg::CuSolutionWeightedAverage 
    step_size::Float64
    primal_weight::Float64
    numerical_error::Bool
    cumulative_kkt_passes::Float64
    total_number_iterations::Int64
    CG_total_extra::Int64
    required_ratio::Union{Float64,Nothing}
    ratio_step_sizes::Union{Float64,Nothing}
    l2_norm_lorank_obj_matrix::Float64
    l2_norm_constraint_matrix::Float64
    current_gradient::CuVector{Float64}
    current_direction::CuVector{Float64}
    CG_switch::Bool

end
mutable struct QP_constant_paramter_gpu
    Q_origin::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64,Int32}
    Q_scaled::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64,Int32}
end
mutable struct CuBufferState
    next_primal::CuVector{Float64}
    next_dual::CuVector{Float64}
    delta_primal::CuVector{Float64}
    delta_dual::CuVector{Float64}
    next_primal_product::CuVector{Float64}
    delta_primal_product::CuVector{Float64}
    next_dual_product::CuVector{Float64}
    delta_dual_product::CuVector{Float64}
    next_primal_obj_product::CuVector{Float64} 
    delta_primal_obj_product::CuVector{Float64} 
    CG_product::CuVector{Float64} 
    CG_bound::Float64
end

function Pdhcg_specific_log(
    iteration::Int64,
    current_primal_solution::CuVector{Float64},
    current_dual_solution::CuVector{Float64},
    step_size::Float64,
    required_ratio::Union{Float64,Nothing},
    primal_weight::Float64,
)
    Printf.@printf(
        "   %5d norms=(%9g, %9g) inv_step_size=%9g ",
        iteration,
        norm(current_primal_solution),
        norm(current_dual_solution),
        1 / step_size,
    )
    if !isnothing(required_ratio)
        Printf.@printf(
        "   primal_weight=%18g  inverse_ss=%18g\n",
        primal_weight,
        required_ratio
        )
    else
        Printf.@printf(
        "   primal_weight=%18g \n",
        primal_weight,
        )
    end
end

"""
Kernel to compute primal solution in the next iteration
"""

function compute_next_primal_solution_kernel_update!(
    next_primal::CuDeviceVector{Float64},
    current_gradient::CuDeviceVector{Float64},
    current_direction::CuDeviceVector{Float64},
    CG_product::CuDeviceVector{Float64},
    alpha::Float64
)
    tx = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if tx <= length(next_primal)
        @inbounds begin
            next_primal[tx] = next_primal[tx] - alpha * current_direction[tx]
            current_gradient[tx] = current_gradient[tx] - alpha * CG_product[tx]
        end
    end
    return 
end

"""
Compute primal solution in the next iteration
"""
function compute_next_primal_solution!(
    problem::CuQuadraticProgrammingProblem,
    current_primal_solution::CuVector{Float64},
    current_dual_product::CuVector{Float64},
    current_primal_obj_product::CuVector{Float64}, 
    total_iteration::Int64,
    step_size::Float64,
    primal_weight::Float64,
    CG_bound::Float64,
    current_gradient::CuVector{Float64},
    current_direction::CuVector{Float64},
    next_primal::CuVector{Float64},
    CG_product::CuVector{Float64},
    next_primal_product::CuVector{Float64},
    next_primal_obj_product::CuVector{Float64}, 
    first_iter::Bool
)
    max_CG_iter = 20

    current_gradient .= current_primal_obj_product .+ problem.objective_vector .- current_dual_product
    current_direction .= current_gradient
    tmp_rank_vec = CUDA.zeros(Float64, problem.num_rank)
    if first_iter
        alpha = step_size / primal_weight
        next_primal .= next_primal .- (step_size/primal_weight).* current_direction
        CG_iter = 0
        # println("check next primal", any(isnan.(next_primal)))
        # println("check current direction", any(isnan.(current_direction)))
        # println("check current gradient", any(isnan.(current_gradient)))
        # println("check current primal obj product", any(isnan.(current_primal_obj_product)))
        # println("check current dual product", any(isnan.(current_dual_product)))
        # println("check current primal solution", any(isnan.(current_primal_solution)))
        # println("check primal_weight", isnan(primal_weight))
        # println("primal_weight", primal_weight)

    else
        gg = CUDA.dot(current_gradient, current_gradient)

        CG_iter = 1
        next_primal .= current_primal_solution

        while CG_iter <= max_CG_iter
            gkgk = gg
            CUDA.CUSPARSE.mv!('N', 1.0, problem.lorank_obj_matrix, current_direction, 0.0, tmp_rank_vec,'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
            CUDA.CUSPARSE.mv!('T', 1.0, problem.lorank_obj_matrix, tmp_rank_vec, 0.0, CG_product,'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)

            CG_product .+= (primal_weight / step_size + problem.regularization).*current_direction
            dHd = CUDA.dot(current_direction, CG_product)
            alpha = gg / dHd

            next_primal .= next_primal .- alpha.* current_direction
            current_gradient .= current_gradient .- alpha.*CG_product
            gg = CUDA.dot(current_gradient, current_gradient)
            
            if sqrt(gg) <= min(0.05 * CG_bound, 1e-2)
                break
            end  

            current_direction .= (gg /gkgk) .* current_direction .+ current_gradient
            CG_iter += 1
        end
    end

    CUDA.CUSPARSE.mv!('N', 1.0, problem.constraint_matrix, next_primal, 0.0, next_primal_product,'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    if total_iteration % 40 == 1 || first_iter
        CUDA.CUSPARSE.mv!('N', 1.0, problem.lorank_obj_matrix, next_primal, 0.0, tmp_rank_vec,'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('T', 1.0, problem.lorank_obj_matrix, tmp_rank_vec, 0.0, next_primal_obj_product,'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        next_primal_obj_product .+= problem.regularization .* next_primal
    else
        next_primal_obj_product .= current_gradient .- problem.objective_vector .+ current_dual_product .- (primal_weight / step_size).*(next_primal .- current_primal_solution)
    end
    
    CG_iter = min(CG_iter, max_CG_iter)
    return CG_iter
end
function projection!(
    primal::CuArray{Float64, 1}, 
    variable_lower_bound::CuArray{Float64, 1}, 
    variable_upper_bound::CuArray{Float64, 1}
    )
    CUDA.@sync @. primal = clamp(primal, variable_lower_bound, variable_upper_bound)
end

function compute_next_primal_solution_gd_BB!(
    problem::CuQuadraticProgrammingProblem,
    current_primal_solution::CuArray{Float64,1},
    current_dual_product::CuArray{Float64,1},
    current_primal_obj_product::CuArray{Float64,1},
    last_gradient::CuArray{Float64,1},
    step_size::Float64,
    primal_weight::Float64,
    CG_bound::Float64,
    current_gradient::CuArray{Float64,1},
    inner_delta_primal::CuArray{Float64,1},
    next_primal::CuArray{Float64,1},
    last_primal::CuArray{Float64,1},
    next_primal_product::CuArray{Float64,1},
    next_primal_obj_product::CuArray{Float64,1},
    norm_Q::Float64,
    first_iter::Bool
)

    max_CG_iter = 100
    k = 1
    alpha = 1.0 / (norm_Q + primal_weight / step_size)

    CUDA.copyto!(last_primal, current_primal_solution)
    current_gradient .= current_primal_obj_product .+ problem.objective_vector .- current_dual_product
    CUDA.copyto!(last_gradient, current_gradient)
    next_primal .= current_primal_solution .- 1.0 / (norm_Q + primal_weight / step_size) .* current_gradient

    projection!(next_primal, problem.variable_lower_bound, problem.variable_upper_bound)
    tmp_rank_vec = CUDA.zeros(Float64, problem.num_rank)
    while k <= max_CG_iter
        inner_delta_primal .= next_primal .- last_primal

        gg = CUDA.dot(inner_delta_primal, inner_delta_primal)
        if sqrt(gg) <= min(0.05*CG_bound, 1e-2)*alpha
            break
        end
        CUDA.CUSPARSE.mv!('N', 1.0, problem.lorank_obj_matrix, next_primal, 0.0, tmp_rank_vec,'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('T', 1.0, problem.lorank_obj_matrix, tmp_rank_vec, 0.0, current_gradient,'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        current_gradient .= current_gradient .+ (primal_weight / step_size) .* ((1 + problem.regularization) .* next_primal .- current_primal_solution) .+ problem.objective_vector .- current_dual_product
        alpha = gg / CUDA.dot(inner_delta_primal, current_gradient .- last_gradient)

        CUDA.copyto!(last_primal, next_primal)
        CUDA.copyto!(last_gradient, current_gradient)
       
        next_primal .= next_primal .- alpha .* current_gradient
        projection!(next_primal, problem.variable_lower_bound, problem.variable_upper_bound)
        k += 1
    end
    CUDA.CUSPARSE.mv!('N', 1.0, problem.lorank_obj_matrix, next_primal, 0.0, tmp_rank_vec,'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    CUDA.CUSPARSE.mv!('T', 1.0, problem.lorank_obj_matrix, tmp_rank_vec, 0.0, next_primal_obj_product,'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    next_primal_obj_product .+= problem.regularization .* next_primal
    CUDA.CUSPARSE.mv!('N', 1.0, problem.constraint_matrix, next_primal, 0.0, next_primal_product,'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)


    CG_iter = min(k, max_CG_iter)
    return CG_iter
end

"""
Kernel to compute dual solution in the next iteration
"""
function compute_next_dual_solution_kernel!(
    right_hand_side::CuDeviceVector{Float64},
    current_dual_solution::CuDeviceVector{Float64},
    current_primal_product::CuDeviceVector{Float64},
    next_primal_product::CuDeviceVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    num_equalities::Int64,
    num_constraints::Int64,
    next_dual::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_equalities
        @inbounds begin
            next_dual[tx] = current_dual_solution[tx] + (primal_weight * step_size) * (right_hand_side[tx] - next_primal_product[tx])
        end
    elseif (num_equalities + 1) <= tx <= num_constraints
        @inbounds begin
            next_dual[tx] = current_dual_solution[tx] + (primal_weight * step_size) * (right_hand_side[tx] - next_primal_product[tx])
            next_dual[tx] = max(next_dual[tx], 0.0)
        end
    end
    return 
end

"""
Compute dual solution in the next iteration
"""
function compute_next_dual_solution!(
    problem::CuQuadraticProgrammingProblem,
    current_dual_solution::CuVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    next_primal_product::CuVector{Float64},
    current_primal_product::CuVector{Float64},
    next_dual::CuVector{Float64},
    next_dual_product::CuVector{Float64},
)
    NumBlockDual = ceil(Int64, problem.num_constraints/ThreadPerBlock)

    CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockDual compute_next_dual_solution_kernel!(
        problem.right_hand_side,
        current_dual_solution,
        current_primal_product,
        next_primal_product,
        step_size,
        primal_weight,
        problem.num_equalities,
        problem.num_constraints,
        next_dual,
    )

    CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix_t, next_dual, 0, next_dual_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
end

"""
Update primal and dual solutions
"""
function update_solution_in_solver_state!(
    solver_state::CuPdhcgSolverState,
    buffer_state::CuBufferState,
)
    solver_state.delta_primal .= buffer_state.next_primal .- solver_state.current_primal_solution
    solver_state.delta_dual .= buffer_state.next_dual .- solver_state.current_dual_solution
    print("next_primal", buffer_state.next_primal)
    solver_state.current_primal_solution .= copy(buffer_state.next_primal)
    solver_state.current_dual_solution .= copy(buffer_state.next_dual)
    solver_state.current_dual_product .= copy(buffer_state.next_dual_product)
    solver_state.current_primal_product .= copy(buffer_state.next_primal_product)
    solver_state.current_primal_obj_product .= copy(buffer_state.next_primal_obj_product)

    weight = 1 / (1.0 + solver_state.solution_weighted_avg.primal_solutions_count)
    add_to_solution_weighted_average!(
        solver_state.solution_weighted_avg,
        solver_state.current_primal_solution,
        solver_state.current_dual_solution,
        weight,
        solver_state.current_primal_product,
        solver_state.current_dual_product,
        solver_state.current_primal_obj_product,
    )
end
function cal_multiple_dot(
    a::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}, 
    b_vectors::Vector{CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}},
)   
    result = CUDA.zeros(Float64, length(b_vectors))
    b_matrix = CUDA.hcat(b_vectors...)
    CUDA.CUBLAS.gemv!('T', 1.0, b_matrix, a, 0.0, result)
    return result
end

function compute_triple_dots_kernel!(
    delta_primal::CUDA.CuDeviceVector{Float64, 1},
    delta_dual_product::CUDA.CuDeviceVector{Float64, 1},
    delta_primal_obj_product::CUDA.CuDeviceVector{Float64, 1},
    result::CUDA.CuDeviceVector{Float64, 1})
    
    tx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if tx <= length(delta_primal)
        @inbounds begin
            CUDA.atomic_add!(pointer(result, 1), delta_primal[tx] * delta_dual_product[tx])
            CUDA.atomic_add!(pointer(result, 2), delta_primal[tx] * delta_primal_obj_product[tx])
            CUDA.atomic_add!(pointer(result, 3), delta_primal[tx] * delta_primal[tx])
        end
    end
    return
end

function compute_interaction_and_movement(
    solver_state::CuPdhcgSolverState,
    buffer_state::CuBufferState,
)
    buffer_state.delta_primal .= buffer_state.next_primal .- solver_state.current_primal_solution
    buffer_state.delta_dual .= buffer_state.next_dual .- solver_state.current_dual_solution
    buffer_state.delta_dual_product .= buffer_state.next_dual_product .- solver_state.current_dual_product
    buffer_state.delta_primal_product .= buffer_state.next_primal_product .- solver_state.current_primal_product
    buffer_state.delta_primal_obj_product .= buffer_state.next_primal_obj_product .- solver_state.current_primal_obj_product
    
    primal_dual_interaction = CUDA.dot(buffer_state.delta_primal, buffer_state.delta_dual_product)
    primal_objective_interaction = CUDA.dot(buffer_state.delta_primal, buffer_state.delta_primal_obj_product)
    norm_delta_primal = CUDA.dot(buffer_state.delta_primal, buffer_state.delta_primal)

    interaction = abs.(primal_dual_interaction)+0.5*abs.(primal_objective_interaction) 
    norm_delta_dual = CUDA.dot(buffer_state.delta_dual,buffer_state.delta_dual)

    movement = 0.5 * solver_state.primal_weight * norm_delta_primal + (0.5 / solver_state.primal_weight) * norm_delta_dual
    return interaction, movement
end
"""
Take PDHG step with ConstantStepsize
"""
function take_step!(
    step_params::ConstantStepsizeParams,
    problem::CuQuadraticProgrammingProblem,
    solver_state::CuPdhcgSolverState,
    buffer_state::CuBufferState,
)
    step_size = solver_state.step_size
    done = false
    k = 1
    while !done
        k += 1
        if k>=20
            break
        end
        solver_state.total_number_iterations += 1
        first_iter = false

        if solver_state.total_number_iterations <= 1
            first_iter = true
        end

    if solver_state.CG_switch
        CG_extra = compute_next_primal_solution!(
            problem,
            solver_state.current_primal_solution,
            solver_state.current_dual_product,
            solver_state.current_primal_obj_product,
            solver_state.total_number_iterations,
            step_size,
            solver_state.primal_weight,
            buffer_state.CG_bound,
            solver_state.current_gradient,
            solver_state.current_direction,
            buffer_state.next_primal,
            buffer_state.CG_product,
            buffer_state.next_primal_product,
            buffer_state.next_primal_obj_product,
            first_iter,
        )
    else
        CG_extra = compute_next_primal_solution_gd_BB!(
            problem,
            solver_state.current_primal_solution,
            solver_state.current_dual_product,
            solver_state.current_primal_obj_product,
            buffer_state.delta_primal_obj_product,
            step_size,
            solver_state.primal_weight,
            buffer_state.CG_bound,
            solver_state.current_gradient,
            solver_state.current_direction,
            buffer_state.next_primal,
            buffer_state.CG_product,
            buffer_state.next_primal_product,
            buffer_state.next_primal_obj_product,
            solver_state.l2_norm_lorank_obj_matrix,
            first_iter,
        )
    end
    solver_state.CG_total_extra += CG_extra
    
    compute_next_dual_solution!(
        problem,
        solver_state.current_dual_solution,
        solver_state.step_size,
        solver_state.primal_weight,
        buffer_state.next_primal_product,
        solver_state.current_primal_product,
        buffer_state.next_dual,
        buffer_state.next_dual_product,
    )

    solver_state.cumulative_kkt_passes += 1
    interaction, movement = compute_interaction_and_movement(
        solver_state,
        buffer_state,
    )


    if interaction > 0
        step_size_limit = movement / interaction
        if movement == 0.0
            # The algorithm will terminate at the beginning of the next iteration
            solver_state.numerical_error = true
            break
        end
    else
        step_size_limit = Inf
    end

    if step_size <= step_size_limit
        update_solution_in_solver_state!(
            solver_state, 
            buffer_state,
        )
        done = true
    end
    
    first_term = (1 - 1/(solver_state.total_number_iterations + 1)^(0.3)) * step_size_limit
    second_term = (1 + 1/(solver_state.total_number_iterations + 1)^(0.6)) * step_size
    step_size = min(first_term, second_term)
    
    
end

    solver_state.step_size = step_size


end


"""
Main algorithm
"""
function optimize_gpu(
    params::PdhcgParameters,
    original_problem::QuadraticProgrammingProblem,
)
    validate(original_problem)
    qp_cache = cached_quadratic_program_info(original_problem)
    stopkkt_Q = true  #true means stop kkt is calculate by Q
    empty_lb_inf = isempty(findall(original_problem.variable_lower_bound.>-Inf))
    empty_ub_inf = isempty(findall(original_problem.variable_upper_bound.<Inf))
    CG_switch = empty_lb_inf && empty_ub_inf
    #println("begin to Power method")#FIXME
    #println("objective matrix size", size(original_problem.objective_matrix))
    #println("constraint matrix size", size(original_problem.constraint_matrix))
    #println("Sparsity of Objective Matrix", nnz(original_problem.objective_matrix)/(size(original_problem.objective_matrix)[1]*size(original_problem.objective_matrix)[2]))
    #println("Sparsity of Constraint Matrix", nnz(original_problem.constraint_matrix)/(size(original_problem.constraint_matrix)[1]*size(original_problem.constraint_matrix)[2]))
    #println("begin to rescale")#FIXME
    scaled_problem = rescale_problem(
        params.l_inf_ruiz_iterations,
        params.l2_norm_rescaling,
        params.pock_chambolle_alpha,
        params.verbosity,
        original_problem,
    )
    #println(" begin to QP_constant")#FIXME
    if stopkkt_Q 

        dim = length(scaled_problem.variable_rescaling)
        rowval = collect(1:dim)
        colptr = collect(1:dim+1)
        nzval = 1 ./ scaled_problem.variable_rescaling
        diag_tmp = SparseMatrixCSC(dim, dim, colptr, rowval, nzval)
        scaled_P = original_problem.lorank_obj_matrix * diag_tmp

        d_lorank_obj_matrix = CUDA.CUSPARSE.CuSparseMatrixCSR(original_problem.lorank_obj_matrix)
        d_scaled_P = CUDA.CUSPARSE.CuSparseMatrixCSR(scaled_P)
        QP_constant = QP_constant_paramter_gpu(d_lorank_obj_matrix , d_scaled_P)
    
    end
    #println("begin to size")#FIXME
    primal_size = length(scaled_problem.scaled_qp.variable_lower_bound)
    dual_size = length(scaled_problem.scaled_qp.right_hand_side)
    num_eq = scaled_problem.scaled_qp.num_equalities
    if params.primal_importance <= 0 || !isfinite(params.primal_importance)
        error("primal_importance must be positive and finite")
    end

    #println("begin to GPU")#FIXME
    d_scaled_problem = scaledqp_cpu_to_gpu(scaled_problem)
    d_problem = d_scaled_problem.scaled_qp

    #println("begin to Power method agin")#FIXME
    norm_P, number_of_power_iterations_P = estimate_maximum_singular_value(scaled_problem.scaled_qp.lorank_obj_matrix)
    norm_A, number_of_power_iterations_A = estimate_maximum_singular_value(scaled_problem.scaled_qp.constraint_matrix)

    #println("begin to initialization")#FIXME
    # initialization
    solver_state = CuPdhcgSolverState(
        CUDA.zeros(Float64, primal_size),    # current_primal_solution
        CUDA.zeros(Float64, dual_size),      # current_dual_solution
        CUDA.zeros(Float64, primal_size),    # delta_primal
        CUDA.zeros(Float64, dual_size),      # delta_dual
        CUDA.zeros(Float64, dual_size),      # current_primal_product
        CUDA.zeros(Float64, primal_size),    # current_dual_product
        CUDA.zeros(Float64, primal_size),    # current_primal_obj_product
        cu_initialize_solution_weighted_average(primal_size, dual_size),
        0.0,                 # step_size
        1.0,                 # primal_weight
        false,               # numerical_error
        0.0,                 # cumulative_kkt_passes
        0,                   # total_number_iterations
        0,                   # CG_number_iterations
        nothing,
        nothing,
        norm_P ^ 2 + scaled_problem.scaled_qp.regularization,
        norm_A,
        CUDA.zeros(Float64, primal_size),    # current_gradient
        CUDA.zeros(Float64, primal_size),    # current_direction
        CG_switch,
        )

    buffer_state = CuBufferState(
        CUDA.zeros(Float64, primal_size),      # next_primal
        CUDA.zeros(Float64, dual_size),        # next_dual
        CUDA.zeros(Float64, primal_size),      # delta_primal
        CUDA.zeros(Float64, dual_size),        # delta_dual
        CUDA.zeros(Float64, dual_size),        # next_primal_product
        CUDA.zeros(Float64, dual_size),        # delta_primal_product
        CUDA.zeros(Float64, primal_size),      # next_dual_product
        CUDA.zeros(Float64, primal_size),      # delta_dual_product
        CUDA.zeros(Float64, primal_size),      # next_primal_obj_product
        CUDA.zeros(Float64, primal_size),      # delta_next_primal_obj_product
        CUDA.zeros(Float64, primal_size),      # CG_product,
        1e-3,                                  # CG_bound
    )

    buffer_avg = CuBufferAvgState(
        CUDA.zeros(Float64, primal_size),      # avg_primal_solution
        CUDA.zeros(Float64, dual_size),        # avg_dual_solution
        CUDA.zeros(Float64, dual_size),        # avg_primal_product
        CUDA.zeros(Float64, primal_size),      # avg_dual_product
        CUDA.zeros(Float64, primal_size),      # avg_primal_gradient
        CUDA.zeros(Float64, primal_size),      # avg_primal_obj_product
    )

    buffer_original = CuBufferOriginalSol(
        CUDA.zeros(Float64, primal_size),      # primal
        CUDA.zeros(Float64, dual_size),        # dual
        CUDA.zeros(Float64, dual_size),        # primal_product
        CUDA.zeros(Float64, primal_size),      # dual_product
        CUDA.zeros(Float64, primal_size),      # primal_gradient
        CUDA.zeros(Float64, primal_size),      # primal_obj_product
    )

    buffer_kkt = CuBufferKKTState(
        CUDA.zeros(Float64, primal_size),      # primal
        CUDA.zeros(Float64, dual_size),        # dual
        CUDA.zeros(Float64, dual_size),        # primal_product
        CUDA.zeros(Float64, primal_size),      # primal_gradient
        CUDA.zeros(Float64, primal_size),      # primal_obj_product
        CUDA.zeros(Float64, primal_size),      # lower_variable_violation
        CUDA.zeros(Float64, primal_size),      # upper_variable_violation
        CUDA.zeros(Float64, dual_size),        # constraint_violation
        CUDA.zeros(Float64, primal_size),      # dual_objective_contribution_array
        CUDA.zeros(Float64, primal_size),      # reduced_costs_violations
        CuDualStats(
            0.0,
            CUDA.zeros(Float64, dual_size - num_eq),
            CUDA.zeros(Float64, primal_size),
        ),
        0.0,                                   # dual_res_inf
    )
    
    buffer_primal_gradient = CUDA.zeros(Float64, primal_size)
    buffer_primal_gradient .= d_scaled_problem.scaled_qp.objective_vector .- solver_state.current_dual_product
    buffer_primal_gradient .+= solver_state.current_primal_obj_product
    current_primal_obj_product_Q = CUDA.zeros(Float64, primal_size)
    avg_primal_obj_product_Q = CUDA.zeros(Float64, primal_size)
    avg_primal_gradient = CUDA.zeros(Float64, primal_size)

    solver_state.cumulative_kkt_passes += number_of_power_iterations_P + number_of_power_iterations_A

    if params.scale_invariant_initial_primal_weight
        solver_state.primal_weight = select_initial_primal_weight(
            scaled_problem.scaled_qp,
            1.0,
            1.0,
            params.primal_importance,
            params.verbosity,
        )
    else
        solver_state.primal_weight = params.primal_importance
    end

    solver_state.step_size = 1/norm_A

    KKT_PASSES_PER_TERMINATION_EVALUATION = 2.0

    primal_weight_update_smoothing = params.restart_params.primal_weight_update_smoothing 

    iteration_stats = IterationStats[]
    start_time = time()
    time_spent_doing_basic_algorithm = 0.0

    last_restart_info = create_last_restart_info(
        d_scaled_problem.scaled_qp,
        solver_state.current_primal_solution,
        solver_state.current_dual_solution,
        solver_state.current_primal_product,
        solver_state.current_dual_product,
        buffer_primal_gradient,
        solver_state.current_primal_obj_product,
    )

    # For termination criteria:
    termination_criteria = params.termination_criteria
    iteration_limit = termination_criteria.iteration_limit
    termination_evaluation_frequency = params.termination_evaluation_frequency

    # This flag represents whether a numerical error occurred during the algorithm
    # if it is set to true it will trigger the algorithm to terminate.
    solver_state.numerical_error = false
    display_iteration_stats_heading()

    
    #println("Enter Loop")#FIXME
    iteration = 0
    while true
        iteration += 1

        if mod(iteration - 1, termination_evaluation_frequency) == 0 ||
            iteration == iteration_limit + 1 ||
            iteration <= 10 ||
            solver_state.numerical_error
            
            solver_state.cumulative_kkt_passes += KKT_PASSES_PER_TERMINATION_EVALUATION

            ### average ###
            if solver_state.numerical_error || solver_state.solution_weighted_avg.primal_solutions_count == 0 || solver_state.solution_weighted_avg.dual_solutions_count == 0
                buffer_avg.avg_primal_solution .= solver_state.current_primal_solution
                buffer_avg.avg_dual_solution .= solver_state.current_dual_solution
                buffer_avg.avg_primal_product .= solver_state.current_primal_product
                buffer_avg.avg_dual_product .= solver_state.current_dual_product
                buffer_avg.avg_primal_gradient .= buffer_primal_gradient
                buffer_avg.avg_primal_obj_product .= solver_state.current_primal_obj_product
            else
                compute_average!(solver_state.solution_weighted_avg, buffer_avg, d_problem)
            end
            #println("compute KKT")#FIXME
            ### KKT ###
            if stopkkt_Q
                tmp_rank_vec = CUDA.zeros(Float64, original_problem.num_rank)
                CUDA.CUSPARSE.mv!('N', 1.0, QP_constant.Q_scaled, solver_state.current_primal_solution, 0.0, tmp_rank_vec,'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
                CUDA.CUSPARSE.mv!('T', 1.0, QP_constant.Q_scaled, tmp_rank_vec, 0.0, current_primal_obj_product_Q,'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
                CUDA.CUSPARSE.mv!('N', 1.0, QP_constant.Q_scaled, buffer_avg.avg_primal_solution, 0.0, tmp_rank_vec,'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
                CUDA.CUSPARSE.mv!('T', 1.0, QP_constant.Q_scaled, tmp_rank_vec, 0.0, avg_primal_obj_product_Q,'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)

                buffer_primal_gradient .= d_scaled_problem.scaled_qp.objective_vector .- solver_state.current_dual_product
                buffer_primal_gradient .+= current_primal_obj_product_Q
                avg_primal_gradient .= d_scaled_problem.scaled_qp.objective_vector .- buffer_avg.avg_dual_product .+ avg_primal_obj_product_Q

                current_iteration_stats_avg = evaluate_unscaled_iteration_stats(
                d_scaled_problem,
                qp_cache,
                params.termination_criteria,
                params.record_iteration_stats,
                buffer_avg.avg_primal_solution,
                buffer_avg.avg_dual_solution,
                iteration,
                time() - start_time,
                solver_state.cumulative_kkt_passes,
                termination_criteria.eps_optimal_absolute,
                termination_criteria.eps_optimal_relative,
                solver_state.step_size,
                solver_state.primal_weight,
                POINT_TYPE_AVERAGE_ITERATE, 
                buffer_avg.avg_primal_product,
                buffer_avg.avg_dual_product,
                avg_primal_gradient,
                avg_primal_obj_product_Q,
                buffer_original,
                buffer_kkt,
            )
            current_iteration_stats_current = evaluate_unscaled_iteration_stats(
                d_scaled_problem,
                qp_cache,
                params.termination_criteria,
                params.record_iteration_stats,
                solver_state.current_primal_solution,
                solver_state.current_dual_solution,
                iteration,
                time() - start_time,
                solver_state.cumulative_kkt_passes,
                termination_criteria.eps_optimal_absolute,
                termination_criteria.eps_optimal_relative,
                solver_state.step_size,
                solver_state.primal_weight,
                POINT_TYPE_AVERAGE_ITERATE, 
                solver_state.current_primal_product,
                solver_state.current_dual_product,
                buffer_primal_gradient,
                current_primal_obj_product_Q,
                buffer_original,
                buffer_kkt,
            )
            else
                current_iteration_stats_avg = evaluate_unscaled_iteration_stats(
                d_scaled_problem,
                qp_cache,
                params.termination_criteria,
                params.record_iteration_stats,
                buffer_avg.avg_primal_solution,
                buffer_avg.avg_dual_solution,
                iteration,
                time() - start_time,
                solver_state.cumulative_kkt_passes,
                termination_criteria.eps_optimal_absolute,
                termination_criteria.eps_optimal_relative,
                solver_state.step_size,
                solver_state.primal_weight,
                POINT_TYPE_AVERAGE_ITERATE, 
                buffer_avg.avg_primal_product,
                buffer_avg.avg_dual_product,
                buffer_avg.avg_primal_gradient,
                buffer_avg.avg_primal_obj_product,
                buffer_original,
                buffer_kkt,
            )
            current_iteration_stats_current = evaluate_unscaled_iteration_stats(
                d_scaled_problem,
                qp_cache,
                params.termination_criteria,
                params.record_iteration_stats,
                solver_state.current_primal_solution,
                solver_state.current_dual_solution,
                iteration,
                time() - start_time,
                solver_state.cumulative_kkt_passes,
                termination_criteria.eps_optimal_absolute,
                termination_criteria.eps_optimal_relative,
                solver_state.step_size,
                solver_state.primal_weight,
                POINT_TYPE_AVERAGE_ITERATE, 
                solver_state.current_primal_product,
                solver_state.current_dual_product,
                buffer_primal_gradient,
                solver_state.current_primal_obj_product,
                buffer_original,
                buffer_kkt,
            )
            end
            c_i_current = current_iteration_stats_current.convergence_information[1]
            c_i_avg = current_iteration_stats_avg.convergence_information[1]

            current_kkt_err = max(c_i_current.relative_optimality_gap, c_i_current.relative_l_inf_primal_residual,c_i_current.relative_l_inf_dual_residual)

            avg_kkt_err = max(c_i_avg.relative_optimality_gap,c_i_avg.relative_l_inf_primal_residual, c_i_avg.relative_l_inf_dual_residual)

            if current_kkt_err >= avg_kkt_err
                current_iteration_stats = current_iteration_stats_avg
                kkt_err = avg_kkt_err
                else
                current_iteration_stats = current_iteration_stats_current
                kkt_err = current_kkt_err
            end

            
            method_specific_stats = current_iteration_stats.method_specific_stats
            method_specific_stats["time_spent_doing_basic_algorithm"] =
                time_spent_doing_basic_algorithm

            primal_norm_params, dual_norm_params = define_norms(
                primal_size,
                dual_size,
                solver_state.step_size,
                solver_state.primal_weight,
            )
            
            ### check termination criteria ###
            termination_reason = check_termination_criteria(
                termination_criteria,
                qp_cache,
                current_iteration_stats,
            )
            if solver_state.numerical_error && termination_reason == false
                termination_reason = TERMINATION_REASON_NUMERICAL_ERROR
            end

            # If we're terminating, record the iteration stats to provide final
            # solution stats.
            if params.record_iteration_stats || termination_reason != false
                push!(iteration_stats, current_iteration_stats)
            end

            # Print table.
            if print_to_screen_this_iteration(
                termination_reason,
                iteration,
                params.verbosity,
                termination_evaluation_frequency,
            )
                display_iteration_stats(current_iteration_stats)
            end

            if termination_reason != false
                # ** Terminate the algorithm **
                # This is the only place the algorithm can terminate. Please keep it this way.
                
                avg_primal_solution = zeros(primal_size)
                avg_dual_solution = zeros(dual_size)
                gpu_to_cpu!(
                    buffer_avg.avg_primal_solution,
                    buffer_avg.avg_dual_solution,
                    avg_primal_solution,
                    avg_dual_solution,
                )

                Pdhcg_final_log(
                    scaled_problem.scaled_qp,
                    avg_primal_solution,
                    avg_dual_solution,
                    params.verbosity,
                    iteration,
                    solver_state.CG_total_extra,
                    termination_reason,
                    current_iteration_stats,
                )

                return unscaled_saddle_point_output(
                    scaled_problem,
                    avg_primal_solution,
                    avg_dual_solution,
                    termination_reason,
                    iteration - 1,
                    iteration_stats,
                    solver_state.CG_total_extra,
                )
            end

            buffer_primal_gradient .= d_scaled_problem.scaled_qp.objective_vector .- solver_state.current_dual_product
            buffer_primal_gradient .+= solver_state.current_primal_obj_product

            current_iteration_stats.restart_used = run_restart_scheme(
                d_scaled_problem.scaled_qp,
                solver_state.solution_weighted_avg,
                solver_state.current_primal_solution,
                solver_state.current_dual_solution,
                last_restart_info,
                iteration - 1,
                primal_norm_params,
                dual_norm_params,
                solver_state.primal_weight,
                params.verbosity,
                params.restart_params,
                solver_state.current_primal_product,
                solver_state.current_dual_product,
                buffer_avg,
                buffer_kkt,
                buffer_primal_gradient,
                solver_state.current_primal_obj_product,
            )
            flag_update_CG_bound = false
            if current_iteration_stats.restart_used != RESTART_CHOICE_NO_RESTART
                flag_update_CG_bound = true
                solver_state.primal_weight = compute_new_primal_weight(
                    last_restart_info,
                    solver_state.primal_weight,
                    primal_weight_update_smoothing,
                    params.verbosity,
                )

            end

            if flag_update_CG_bound
                buffer_state.CG_bound = kkt_err
            else
                buffer_state.CG_bound += kkt_err
            end

        end

        time_spent_doing_basic_algorithm_checkpoint = time()
      
        if params.verbosity >= 6 && print_to_screen_this_iteration(
            false, # termination_reason
            iteration,
            params.verbosity,
            termination_evaluation_frequency,
        )
            Pdhcg_specific_log(
                iteration,
                solver_state.current_primal_solution,
                solver_state.current_dual_solution,
                solver_state.step_size,
                solver_state.required_ratio,
                solver_state.primal_weight,
            )
          end

        take_step!(params.step_size_policy_params, d_problem, solver_state, buffer_state)

        time_spent_doing_basic_algorithm += time() - time_spent_doing_basic_algorithm_checkpoint
    end
end