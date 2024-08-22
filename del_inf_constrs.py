from gurobipy import read, GRB
import os
import coptpy
folder = 'QPLIB2/'
save_folder = 'qplib2/'
for i in os.listdir(folder):
    # 读取模型
    model = read(folder + i)
    model.setParam('Presolve', 0) 

    constraints_to_remove = []
    for constr in model.getConstrs():
        if model.getRow(constr).size() == 0 and constr.Sense == GRB.EQUAL:
            if constr.RHS == 0 and (constr.LB == -GRB.INFINITY and constr.UB == GRB.INFINITY):
                constraints_to_remove.append(constr)

    for constr in constraints_to_remove:
        model.remove(constr)

    model.update()

    model.write(save_folder+i)
    copt_env = coptpy.Envr()
    model_copt = copt_env.createModel()
    model_copt.read(save_folder+i)
    model_copt.write(save_folder+i)