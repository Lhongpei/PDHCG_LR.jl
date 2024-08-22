function generate_randomQP_problem(n::Int, seed::Int=1)
    Random.seed!(seed)
    m = Int(0.5 * n)
    r = 1000
    # Generate problem data
    P = sprandn(r, n, 1e-4)
    rowval = collect(1:n)
    colptr = collect(1:n+1)
    nzval = ones(n)
    PtP = P' * P + 1e-2 * SparseMatrixCSC(n, n, colptr, rowval, nzval)
    #PtP = P' * P
    q = randn(n)
    A = sprandn(m, n, 1e-4)

    v = randn(n)   # Fictitious solution
    delta = rand(m)  # To get inequality
    ru = A * v + delta
    rl = -Inf * ones(m)
    lb = -Inf * ones(n)
    ub = Inf * ones(n)
     
    return QuadraticProgrammingProblem(
        size(A, 2),
        size(A, 1),
        lb,
        ub,
        Vector{Bool}(isfinite.(lb)),
        Vector{Bool}(isfinite.(ub)),
        PtP,
        q,
        0.0,
        -A,
        -A',
        -ru,
        0,
    )
end

function generate_randomQP_eq_problem(n::Int, seed::Int=1)
    Random.seed!(seed)
    m = Int(0.5 * n)
    r = 1000
    # Generate problem data
    P = sprandn(r, n, 1e-4)
    rowval = collect(1:n)
    colptr = collect(1:n+1)
    nzval = ones(n)
    PtP = P' * P + 1e-02 * SparseMatrixCSC(n, n, colptr, rowval, nzval)
    #PtP = P' * P
    q = randn(n)
    A = sprandn(m, n, 1e-4)

    v = randn(n)   # Fictitious solution
    #delta = rand(m)  # To get inequality
    ru = A * v
    #rl = -Inf * ones(m)
    lb = -Inf * ones(n)
    ub = Inf * ones(n)
     
    return QuadraticProgrammingProblem(
        size(A, 2),
        size(A, 1),
        lb,
        ub,
        Vector{Bool}(isfinite.(lb)),
        Vector{Bool}(isfinite.(ub)),
        PtP,
        q,
        0.0,
        A,
        A',
        ru,
        m,
    )
end

function generate_lasso_problem(n::Int, seed::Int=1)
    # Set random seed
    Random.seed!(seed)

    # Initialize parameters
    m = Int(n * 0.5)
    Ad = sprandn(m, n, 1e-4)
    x_true = (rand(n) .> 0.5) .* randn(n) ./ sqrt(n)
    bd = Ad * x_true + randn(m)
    lambda_max = norm(Ad' * bd, Inf)
    lambda_param = (1/5) * lambda_max

    # Construct the QP problem
    rowval_m = collect(1:m)
    colptr_m = collect(1:m+1)
    nzval_m = ones(m)
    P = blockdiag(spzeros(n, n), SparseMatrixCSC(m, m, colptr_m, rowval_m, nzval_m .* 2), spzeros(n, n))
    q = vcat(zeros(m + n), lambda_param * ones(n))
    rowval_n = collect(1:n)
    colptr_n = collect(1:n+1)
    nzval_n = ones(n)
    In = SparseMatrixCSC(n, n, colptr_n, rowval_n, nzval_n)
    Onm = spzeros(n, m)
    A = vcat(hcat(Ad, -SparseMatrixCSC(m, m, colptr_m, rowval_m, nzval_m), spzeros(m, n)),
             hcat(In, Onm, -In),
             hcat(-In, Onm, -In))
    rl = vcat(bd, -Inf * ones(n), -Inf * ones(n))
    ru = vcat(bd, zeros(n), zeros(n))
    lb = -Inf * ones(2*n+m)
    ub = Inf * ones(2*n+m)

    return QuadraticProgrammingProblem(
        size(A, 2),
        size(A, 1),
        lb,
        ub,
        Vector{Bool}(isfinite.(lb)),
        Vector{Bool}(isfinite.(ub)),
        P,
        q,
        0.0,
        -A,
        -A',
        -ru,
        m,
    )
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

    #println("norm_A")
    #println(norm(A))
    return QuadraticProgrammingProblem(
        size(A, 2),
        size(A, 1),
        lb,
        ub,
        Vector{Bool}(isfinite.(lb)),
        Vector{Bool}(isfinite.(ub)),
        P,
        q,
        0.0,
        A,
        A',
        ru,
        0,
    )
end

function generate_portfolio_problem(n::Int, seed::Int=1)
    Random.seed!(seed)
    
    n_assets = n 
    k = Int(n*1)
    F = sprandn(n_assets, k, 1e-4)
    D = spdiagm(0 => rand(n_assets) .* sqrt(k))
    mu = randn(n_assets)
    gamma = 1.0

    # Generate QP problem
    rowval1 = collect(1:n_assets)
    colptr1 = collect(1:n_assets + 1)
    nzval1 = rand(n_assets) .* sqrt(k) .* 2

    rowval2 = collect(n_assets + 1:k + n_assets)
    colptr2 = collect(n_assets + 2:k + n_assets + 1)
    nzval2 = ones(k) .* 2

    rowval = vcat(rowval1, rowval2)
    colptr = vcat(colptr1, colptr2)
    nzval = vcat(nzval1, nzval2)

    rand(n_assets) .* sqrt(k)

    rowval_k = collect(1:k)
    colptr_k = collect(1:k + 1)
    nzval_k = ones(k)

    P = SparseMatrixCSC(n_assets + k, n_assets + k, colptr, rowval, nzval)
    q = vcat(-mu ./ gamma, zeros(k))
    A = vcat(
        hcat(sparse(ones(1, n_assets)), spzeros(1, k)),
        hcat(F', -SparseMatrixCSC(k, k, colptr_k, rowval_k, nzval_k)),
    )
    ru = vcat(1.0, zeros(k))

    lb = vcat(zeros(n_assets), -Inf * ones(k))
    ub = vcat(ones(n_assets), Inf * ones(k))

    return QuadraticProgrammingProblem(
        size(A, 2),
        size(A, 1),
        lb,
        ub,
        Vector{Bool}(isfinite.(lb)),
        Vector{Bool}(isfinite.(ub)),
        P,
        q,
        0.0,
        -A,
        -A',
        -ru,
        k+1,
    )
end

function generate_Huber_problem(n::Int, seed::Int=1)
    # Set random seed
    Random.seed!(seed)

    # Initialize properties
    m_data = Int(n * 0.5)        # Number of data-points

    Ad = sprandn(m_data, n, 1e-4)
    x_true = randn(n) / sqrt(n)
    ind95 = rand(m_data) .< 0.95
    bd = Ad * x_true .+ 0.5 .* randn(m_data) .* ind95 .+ 10.0 .* rand(m_data) .* .!ind95

    rowval_m = collect(1:m_data)
    colptr_m = collect(1:m_data+1)
    nzval_m = ones(m_data)
    Im = SparseMatrixCSC(m_data, m_data, colptr_m, rowval_m, nzval_m)

    P = blockdiag(spzeros(n, n), Im, spzeros(2*m_data, 2*m_data))
    q = vcat(zeros(n + m_data), ones(2*m_data))
    A = hcat(Ad,-Im,-Im,Im)
    ru = bd

    # Bounds
    lb = vcat(fill(-Inf, n + m_data), zeros(2*m_data))
    ub = fill(Inf, n + 3*m_data)
   
    return QuadraticProgrammingProblem(
        size(A, 2),
        size(A, 1),
        lb,
        ub,
        Vector{Bool}(isfinite.(lb)),
        Vector{Bool}(isfinite.(ub)),
        P,
        q,
        0.0,
        A,
        A',
        ru,
        m_data,
    )
end

function generate_svm_problem_real(n::Int, seed::Int=1)
    file_path = "./realdata/svm/a9a.xlsx"
    n_features = 123    

    b = []
    A_index = []
        xf = XLSX.openxlsx(file_path)
        # Iterate through each sheet
        sheetnames = XLSX.sheetnames(xf)
        data = xf[sheetnames[1]]
    
        A_index = data[:]
        b = float.(A_index[:,1])
        m,n = size(A_index)
        A = zeros(Float64,m,n_features)
        for i = 1:m
            for j = 1:n
                if mod(j,2)==0 && !ismissing(A_index[i,j])
                    A[i,A_index[i,j]]=A_index[i,j+1]
                end
            end
        end
        A = SparseMatrixCSC(A)
    # 设置随机种子

    # 初始化属性
    m_data = m    # 数据点数量
    gamma_val = 1.0
    b_svm_val = b


    # 生成 QP 问题
    P = spdiagm(0 => vcat(ones(n_features), zeros(m_data)))
    q = vcat(zeros(n_features), (gamma_val) * ones(m_data))

    rowval1 = collect(1:length(b_svm_val))
    colptr1 = collect(1:length(b_svm_val)+1)
    nzval1 = b_svm_val
    rowval2 = collect(1:m_data)
    colptr2 = collect(1:m_data+1)
    nzval2 = ones(m_data)

    #A1 = SparseMatrixCSC(length(b_svm_val),length(b_svm_val), colptr1, rowval1, nzval1)
    #A2 = SparseMatrixCSC(m_data, m_data, colptr2, rowval2, nzval2)
    A1 = spdiagm(b_svm_val)
    A2 = spdiagm(ones(m_data))
    #println(size())
    A = hcat(-A1 * A, A2)
    ru = ones(m_data)

    lb = vcat(-Inf * ones(n_features), zeros(m_data))
    ub = vcat(Inf * ones(n_features), Inf * ones(m_data))

    println("norm_A")
    println(norm(A))
    return QuadraticProgrammingProblem(
        size(A, 2),
        size(A, 1),
        lb,
        ub,
        Vector{Bool}(isfinite.(lb)),
        Vector{Bool}(isfinite.(ub)),
        P,
        q,
        0.0,
        A,
        A',
        ru,
        0,
    )
end