using JuMP
using SCS
using QPSReader
using CSV
using DataFrames
using SCS_GPU_jll
using JuMP
using SCS

function qpsdata_to_jump(qps::QPSData, accuracy::Float64 = 1e-6, gpu_on::Int = 0)
    # 创建模型和设置优化器属性
    optimizer = optimizer_with_attributes(SCS.Optimizer, "eps_rel" => accuracy, "eps_abs" => 1, "warm_start" => false, "time_limit_secs" => 600, "max_iters" => 100000000)
    model = Model(optimizer)
    print('1')
    if gpu_on == 1
        set_optimizer_attribute(model, "linear_solver", SCS.GpuIndirectSolver)
    else
        set_optimizer_attribute(model, "linear_solver", SCS.IndirectSolver)
    end
    
    # 创建变量
    @variable(model, x[1:qps.nvar], base_name=qps.varnames)
    
    for i in 1:qps.nvar
        if !isnothing(qps.lvar) && isfinite(qps.lvar[i])
            set_lower_bound(x[i], qps.lvar[i])
        end
        if !isnothing(qps.uvar) && isfinite(qps.uvar[i])
            set_upper_bound(x[i], qps.uvar[i])
        end
    end
    
    # 创建目标函数
    @expression(model, linear_term, sum(qps.c[i] * x[i] for i in 1:qps.nvar))
    @expression(model, quadratic_term, 
        sum((qps.qrows[k] == qps.qcols[k] ? qps.qvals[k] * x[qps.qrows[k]] * x[qps.qcols[k]] : 2 * qps.qvals[k] * x[qps.qrows[k]] * x[qps.qcols[k]]) 
            for k in 1:length(qps.qvals)))

    
    if qps.objsense == :max
        @objective(model, Max, linear_term + 0.5 * quadratic_term + qps.c0)
    else
        @objective(model, Min, linear_term + 0.5 * quadratic_term + qps.c0)
    end
    
    # 创建约束
    for i in 1:qps.ncon
        # 使用匿名表达式
        expr = @expression(model, sum(qps.avals[k] * x[qps.acols[k]] for k in 1:length(qps.arows) if qps.arows[k] == i))
        if qps.lcon[i] == qps.ucon[i]
            @constraint(model, expr == qps.lcon[i])
        else
            if isfinite(qps.lcon[i])
                @constraint(model, expr >= qps.lcon[i])
            end
            if isfinite(qps.ucon[i])
                @constraint(model, expr <= qps.ucon[i])
            end
        end
    end
    println("Constant:", qps.c0)
    return model
end
function run_qps(qps_file_path, accuracy::Float64 = 1e-6, gpu_on::Int = 0)
    qps = readqps(qps_file_path, mpsformat=:fixed)
    model = qpsdata_to_jump(qps, accuracy, gpu_on)
    optimize!(model)
    println("objective value: ", objective_value(model))
    return model
end
function main(qps_folder::String, save_file::String, accuracy::Float64, gpu_on::Int)
    results = DataFrame(file=String[], solver_time=Float64[])
    for qps_file_path in readdir(qps_folder, join=true)
        println("Processing file: ", qps_file_path)
        flush(stdout)
        try
            model = run_qps(qps_file_path ,accuracy, gpu_on)
            println("objective value: ", objective_value(model))
            time = MOI.get(model, MOI.SolveTimeSec())
            new_row = DataFrame(file=[qps_file_path], solver_time=[time])
            append!(results, new_row)
            CSV.write(save_file, results)
        catch e
            println("Error processing file $qps_file_path: $e")
            flush(stdout)
        end
    end
end

# Example usage
qps_folder = "qptotal"
save_file = "results_qplib_cpu_1e-6.csv"
acc = 1e-6
GPU_on = 1
GPU_id = 1
ENV["CUDA_VISIBLE_DEVICES"] = "$GPU_id"
main(qps_folder, save_file, acc, GPU_on)

