# ADMM in Python
- simplified version
- just does consensus

# ADMM Iteration
\\[
\begin{align}
x^{k+1} &:= \mbox{prox}_{f}(z^k - u^k)\\
z^{k+1} &:= \mbox{prox}_{g}(x^{k+1} + u^k)\\
u^{k+1} &:= u^k + x^{k+1} - z^{k+1}
\end{align}
\\]

- $x$ is the result of each prox in parallel, stacked colleciton of all local variables
- $z$ is the result of the consensus or sharing fusion
    - in consensus, we have many identical copies in the appropriate entries
    - in sharing, the appropriate subgroups sum to their target value

## General residuals
- $r^{k+1} = x^{k+1} - z^{k+1}$
- $s^{k+1} = -\rho(z^{k+1} - z^{k})$

# Consensus Iteration
\\[
\begin{align}
x^{k+1}_i &:= \mbox{prox}_{f_i}(\bar{x}_i^k - u_i^k)\\
u_i^{k+1} &:= u_i^k + x_i^{k+1} - \bar{x}_i^{k+1}
\end{align}
\\]

- note that in this iteration, $z = \bar{x}$

## Residuals
- $r^{k+1} = x^{k+1} - z^{k+1}$
    - sum over all local copies of each variable, accumulating the difference with the global average of that variable
    - that is, difference of local with global, weighted by how many local copies there are
- $s^{k+1} = -\rho(z^{k+1} - z^{k})$
    - that is, difference between global averages, weighted by how many local variables correspond to the global variable

# dual variable robustness
- some tricks with the dual variables are played
- in consensus, we can simplify the algorithm
if we assume $\bar{u}^0 = 0$ and that forever after, we
have that $\bar{u}^k = 0$. I've found that this can be a bug,
especially when initializing $u$ with something. the fix
is to make sure that the average is always zero as the
algorithm progresses (which should be true mathematically,
but we may want to check that its true numerically, or
after potential human tampering)