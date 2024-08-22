using LinearAlgebra
using BenchmarkTools
using SparseArrays
using Random

# Create a random sparse matrix
n = 100000
seed = 1
Random.seed!(seed)
m = Int(0.5 * n)
r = 10

# Generate problem data
P = sprandn(r, n, 0.01)
rowval = collect(1:n)
colptr = collect(1:n+1)
nzval = ones(n)
PtP = P' * P + 1e-2 * SparseMatrixCSC(n, n, colptr, rowval, nzval)

# Allocate CPU vectors
x = ones(Float64, n)
V1 = zeros(Float64, r)
+
V2 = zeros(Float64, n)
VQ = zeros(Float64, n)

# Function to compute the matrix-vector product PtP * x on CPU
function Q(PtP, x, VQ)
    mul!(VQ, PtP, x) # Equivalent to VQ = PtP * x
end

# Function to compute P * x and then P' * V1 on CPU
function PP(P, x, V1, V2)
    mul!(V1, P, x)   # V1 = P * x
    mul!(x, P', V1)  # x = P' * V1
    x .+= 0.01 * V2  # V2 = V2 + 1e-2 * x
end

# Benchmark the functions
PP_time = @benchmark PP(P, x, V1, V2)
Q_time = @benchmark Q(PtP, x, VQ)

# Print runtime
println("Q 运行时间:")
println(Q_time)
println("PP 运行时间:")
println(PP_time)
