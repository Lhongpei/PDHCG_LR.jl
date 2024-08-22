import gurobipy as gp
from gurobipy import GRB

def analyze_qp_model(model_file):
    # 读取模型
    model = gp.read(model_file)
    
    # 获取模型规模
    num_vars = model.NumVars
    num_constrs = model.NumConstrs
    num_nonzeros = model.NumNZs
    
    # 计算稠密度
    if num_constrs * num_vars > 0:
        density = num_nonzeros / (num_constrs * num_vars)
    else:
        density = 0
    
    # 输出结果
    print(f"Number of variables: {num_vars}")
    print(f"Number of constraints: {num_constrs}")
    print(f"Number of non-zero elements: {num_nonzeros}")
    print(f"Density: {density:.10f}")

    # 检查目标函数中的二次项
    obj = model.getObjective()
    num_q_nonzeros = 0
    if isinstance(obj, gp.QuadExpr):
        for i in range(obj.size()):
            var1 = obj.getVar1(i)
            var2 = obj.getVar2(i)
            coeff = obj.getCoeff(i)
            if coeff != 0:
                num_q_nonzeros += 1

    print(f"Number of non-zero quadratic elements in objective: {num_q_nonzeros}")
    if num_vars * num_vars > 0:
        q_density = num_q_nonzeros / (num_vars * num_vars)
    else:
        q_density = 0
    print(f"Quadratic density in objective: {q_density:.10f}")

    # 检查是否存在二次约束
    num_qconstrs = model.NumQConstrs
    if num_qconstrs > 0:
        num_qconstr_nonzeros = 0
        for qc in model.getQConstrs():
            for i in range(qc.Q.size()):
                var1 = qc.Q.getVar1(i)
                var2 = qc.Q.getVar2(i)
                coeff = qc.Q.getCoeff(i)
                if coeff != 0:
                    num_qconstr_nonzeros += 1

        print(f"Number of quadratic constraints: {num_qconstrs}")
        print(f"Number of non-zero quadratic elements in constraints: {num_qconstr_nonzeros}")
        if num_vars * num_vars > 0:
            qconstr_density = num_qconstr_nonzeros / (num_vars * num_vars)
        else:
            qconstr_density = 0
        print(f"Quadratic density in constraints: {qconstr_density:.10f}")

if __name__ == "__main__":
    model_file = "qplib_tot/QPLIB_9008.mps" 
    analyze_qp_model(model_file)
