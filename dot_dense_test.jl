ENV["CUDA_VISIBLE_DEVICES"] = 2

# 其他 CUDA 操作...
using LinearAlgebra
using BenchmarkTools
using CUDA
using CUDA.CUSPARSE
using SparseArrays
using Random

# 创建一个随机稀疏矩阵
n = 100000
seed = 1
Random.seed!(seed)
m = Int(0.5 * n)
r = 1
P = sprandn(r, n,0.01)
rowval = collect(1:n)
colptr = collect(1:n+1)
nzval = ones(n)
PtP = P' * P + 1e-2 * SparseMatrixCSC(n, n, colptr, rowval, nzval)
PtP = CuSparseMatrixCSR(PtP)
d_PtP = CUDA.CUSPARSE.CuSparseMatrixCSR(PtP)
d_P = CUDA.CUSPARSE.CuSparseMatrixCSR(P)
x = CUDA.ones(Float64,n)
V1 = CUDA.zeros(Float64,r)
V2 = CUDA.zeros(Float64,n)
VQ = CUDA.zeros(Float64,n)

PtP = CuArray(PtP)
P = CuArray(P)
# Function to perform matrix-vector multiplication for PtP * x on GPU
function Q(PtP, x, VQ)
    VQ .= PtP * x
end

# Function to perform P * x and then P' * V1 on GPU
function PP(P, x, V1, V2)
    V1 .= P * x  # V1 = P * x
    # x .=  * V1     # x = P' * V1
    # x .+= 0.01 * V2  # V2 = V2 + 1e-2 * x
end

# Benchmark the functions
PP_time = @benchmark PP(d_P, x, V1, V2)
Q_time = @benchmark Q(d_PtP, x, VQ)

# Print runtime
println("Q 运行时间:")
println(Q_time)
println("PP 运行时间:")
println(PP_time)
