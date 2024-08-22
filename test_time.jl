using LinearAlgebra
using BenchmarkTools
using CUDA
using CUDA.CUSPARSE
using SparseArrays
using Random

# 创建一个随机稀疏矩阵
n = 100
rowval = collect(1:n)
colptr = collect(1:n+1)
nzval = ones(n)
#PtP 0= P' * P + 1e-2 * SparseMatrixCSC(n, n, colptr, rowval, nzval)
r = 1

x = CUDA.ones(Float64,n)
V1 = CUDA.zeros(Float64,r)
V2 = CUDA.zeros(Float64,n)
VQ = CUDA.zeros(Float64,n)



seed = 1
Random.seed!(seed)
m = Int(0.5 * n)
P = sprandn(r, n,0.01)
PtP = P' * P
d_PtP = CUDA.CUSPARSE.CuSparseMatrixCSR(PtP)
d_P = CUDA.CUSPARSE.CuSparseMatrixCSR(P)
function Q(PtP,x,VQ)
    CUDA.CUSPARSE.mv!('N', 1.0, PtP, x, 0.0, VQ,'O')
    #synchronize()       
    return VQ
end
function PP(P,x,V1,V2)

    CUDA.CUSPARSE.mv!('N', 1.0, P, x, 0.0, V1,'O')
    CUDA.CUSPARSE.mv!('T', 1.0,P, V1, 0.0,V2,'O')
    V2 .= V2 .+ 1e-2.*x
    #synchronize()
    return V2
end
@time begin
    println("Q")
end


#result1 = Q(d_PtP, x,VQ)
@time begin
for i = 1:10000
    result1 = Q(d_PtP, x, VQ)
end
end
@time begin
    for i =1:10000
        result2 = PP(d_P,x,V1,V2)
    end
    end
#result2 = PP(d_P,x,V1,V2)

# 打印运行时间
#println("Q 运行时间:")
#println(Q_time)
#println("PP 运行时间:")
#println(PP_time)
#println(norm(result1-result2))