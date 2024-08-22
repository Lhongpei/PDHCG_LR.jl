#-------------Running Parameters--------------
#-----------------GPU setting-----------------
GPU_on = 1                     # 1: use GPU; 0: use CPU
GPU_id = 2                        # The GPU id if there are multiple GPUs
#-----------------Task setting----------------
folder_path = "./qptotal/"            # The folder path of the problems
time_limit = 600                      # The time limit for each problem
relat = 1e-6                       # The relative tolerance for the solver
verb = 6                              # The verbosity of the solver
cpu_thread =  32                     # The number of threads for the CPU
save_path = "test/LP/$GPU_on/1e-6"
#---------------------------------------------
#--------------------END----------------------

file_names = readdir(folder_path)
problem_num = length(file_names)
project_scr = ["--project=scripts", "scripts/solve.jl"]
time_limit = ["--time_sec_limit", "$time_limit"]
verb = ["--verbosity", "$verb"]
out_dir = ["--output_dir", "$save_path"]
relat = ["--tolerance", "$relat"]
gpu_option = GPU_on == 1 ? ["--use_gpu", "1"] : ["--use_gpu", "0"]
if GPU_on == 1
    ENV["CUDA_VISIBLE_DEVICES"] = "$GPU_id"  # Setting GPU id if there are multiple GPUs
end
# Start solving the problems
for (i, file_name) in enumerate(file_names)
    println("Start solving the problem: $i, named: $file_name")
    ins_path = ["--instance_path", joinpath(folder_path, file_name)]
    local_problem = `julia $project_scr $ins_path $out_dir $relat $time_limit $gpu_option`
    println("Running command: ", local_problem)
    try
        run(local_problem)
    catch e
        println("Failed to solve $file_name due to error: $e")
    end
end
