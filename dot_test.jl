using LinearAlgebra
using BenchmarkTools
using CUDA
using CUDA.CUSPARSE
using SparseArrays
using Random

ENV["CUDA_VISIBLE_DEVICES"] = 2
# 创建一个随机稀疏矩阵
n = 100000
seed = 1
Random.seed!(seed)
m = Int(0.5 * n)
r = 100
# Generate problem data
P = sprandn(r, n,0.01)
rowval = collect(1:n)
colptr = collect(1:n+1)
nzval = ones(n)
PtP = P' * P + 1e-2 * SparseMatrixCSC(n, n, colptr, rowval, nzval)

d_PtP = CUDA.CUSPARSE.CuSparseMatrixCSR(PtP)
d_P = CUDA.CUSPARSE.CuSparseMatrixCSR(P)
x = CUDA.ones(Float64,n)
t = 100000
t = CUDA.ones(Float64,t)
V1 = CUDA.zeros(Float64,r)
V2 = CUDA.zeros(Float64,n)
VQ = CUDA.zeros(Float64,n)
# 在CPU上计算每一行的二范数
function Q(PtP,x,VQ)
    CUDA.CUSPARSE.mv!('N', 1.0, PtP, x, 0.0, VQ,'O',CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
end

# 在GPU上计算每一行的二范数
function PP(P,x,V1,V2)

    CUDA.CUSPARSE.mv!('N', 1.0, P, x, 0.0, V1,'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    CUDA.CUSPARSE.mv!('T', 1.0,P, V1, 1e-2, x,'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    return 
end

#result1 = Q(d_PtP, x,VQ)
Q_time = @benchmark Q(d_PtP, x, VQ)
Q_memory = CUDA.@allocated Q(d_PtP, x, VQ)

#result2 = PP(d_P,x,V1,V2)
PP_time = @benchmark PP(d_P,x,V1,V2)
PP_memory = CUDA.@allocated PP(d_P,x,V1,V2)

# 打印运行时间和内存分配
println("Q 运行时间:")
println(Q_time)
println("Q 内存分配:")
println(Q_memory, " bytes")

println("PP 运行时间:")
println(PP_time)
println("PP 内存分配:")
println(PP_memory, " bytes")

# println("d_PtP 内存大小: ", CUDA.device_memsize(d_PtP), " bytes")
# println("P 内存大小: ", CUDA.device_memsize(d_P), " bytes")