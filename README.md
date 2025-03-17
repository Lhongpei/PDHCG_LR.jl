# Low-Rank version of Primal-Dual Hybrid Conjugate Gradient Descent (PDHCG)
## Formulation
This Optimizer is designed to solving problem with the following formulation:

$$
    \min_{x \in \mathbb{R}^n} \quad  \frac{1}{2} x^T PP^T x + c^T x \qquad
\text{s.t.} \quad  A x \leq b
$$

where $P \in \mathbb{R}^{n\times r}, r<n$ 

## Usage
```julia
julia run_generated.jl
```
Adjust the parameter in this file to run different generated problems.

*Remark*:
  - Currently, we haven't develop APIs to run customized problems!
  - You shouldn't use low-rank PDHCG for problems without explicit low-rank structure (explicit given $P$ in $PP^T$ or $P$ can be calculated easily).

## TODO_List
- Implement Julia and Python APIs for using low-rank PDHCG.
