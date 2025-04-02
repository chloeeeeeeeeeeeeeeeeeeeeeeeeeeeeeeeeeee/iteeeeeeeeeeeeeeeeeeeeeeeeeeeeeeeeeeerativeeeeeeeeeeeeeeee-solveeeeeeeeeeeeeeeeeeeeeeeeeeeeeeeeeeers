"""
파일 이름: gmres_script.py
작성자: 글로이정 (Chloe Chung)
작성일: 2025년 3월 20일
설명:
    이 스크립트는 GMRES(Generalized Minimal Residual) 알고리즘을 PyTorch로 구현합니다.
    이 방법은 희소하거나 조건이 안 좋은 선형 시스템 Ax = b를 반복적으로 푸는 데 사용됩니다.

    스크립트는 다양한 스펙트럼 조건을 가진 행렬을 생성하고,
    사용자 정의 GMRES 구현을 통해 수렴 속도와 잔차(residual)를 시각화합니다.

사용 방법:
    - 스크립트를 실행하면 5가지 서로 다른 행렬 스펙트럼에 대한 테스트가 자동으로 수행됩니다.
    - 각 테스트의 반복 횟수와 최종 잔차(norm of residual)가 출력됩니다.
    - 수렴 과정은 로그 스케일 그래프로 시각화됩니다.

의존성:
    - Python 3.x
    - NumPy
    - PyTorch
    - Matplotlib

함수 설명:
    - gmres_custom(A, b, x0=None, ...): PyTorch를 이용해 GMRES 알고리즘을 수행합니다.
    - min_res(Hk, beta): 선형 최소제곱 문제 ||Hk y - βe₁|| 를 풀어 y를 구합니다.
    - stop_crit1(...), stop_crit2(...): 잔차가 충분히 작거나 수렴 조건을 만족하는지 확인합니다.
    - cons_mat_w_spect(n, spectrum): 원하는 스펙트럼(고윳값 분포)을 가진 대칭 행렬을 생성합니다.
    - run_test(n, spectrum, label), run_test_case_e(n): 주어진 조건으로 테스트를 실행하고 잔차를 시각화합니다.
    - res_plot(residuals, label): 수렴 과정을 로그 스케일로 플롯합니다.

버전:
    1.0

########## ENG. TRANSLATION ########## (bc I'm a team player) 
Description:
    This script implements GMRES (Generalized Minimal Residual) via PyTorch.
    GMRES's an iterative method used to solve linear systems of the form Ax = b,
    especially when A is fucking huge, sparse, ∧/∨ ill-conditioned.

    This script generates matrices with different spectral properties ∧ tests
    the custom GMRES implementation for each one, showing how quickly it converges.

How to Use:
    - Just run the script ∧ it’ll automatically run the 5 different test cases with
      different eigenvalue distributions (which are called spectra btw).
    - For each test, it prints how many iterations it took ∧ what the final residual was.
    - It also plots a convergence graph (residual norm vs. iteration) in log scale.

Dependencies:
    - Python 3.x
    - NumPy
    - PyTorch
    - Matplotlib

Function Descriptions:
    - gmres_custom(A, b, x0=None, ...): runs the GMRES algorithm using PyTorch.
    - min_res(Hk, beta): solves a least-squares problem to minimize residuals inside GMRES.
    - stop_crit1(...), stop_crit2(...): decides whether to stop based on error size, breakdown, stagnation, or max iter. The difference between the two is that 1 is fixed ∧ 2 is adjustable
    - cons_mat_w_spect(n, spectrum): creates a symmetric matrix with the desired eigenvalues.
    - run_test(n, spectrum, label), run_test_case_e(n): runs test cases ∧ plots the results.
    - res_plot(residuals, label): plots the convergence of GMRES using log-scale residual norms.
"""



###### READ ME ######
#ngl I was mad lazy ∧ did this in sprints so there are like 90 billion
#files but heyyy it's a rough draft for a reason; we can consolidate later
#
#check the corresponding notes for a more comprehensive breakdown
#
#I mentioned it last time but just in case y'all forgot ∧ didn't take discrete
#I use logical ∧ (∧) ∧ logical or (∨) rather than writing them out
#
#LASTLY at the bottom there's a "what am I looking at" section that gives
#a CURSORY rundown of the functions


import numpy as np
import torch
import matplotlib.pyplot as mpl


def min_res(Hk, beta):
    m, n = Hk.shape
    e1 = torch.zeros((m,), dtype=torch.float64)
    e1[0] = beta
    y = torch.linalg.lstsq(Hk, e1.unsqueeze(1)).solution[:n]
    return y.squeeze()

#ask:: which version is preferred?
#checks:
#i. absolute residual norm: "how far from actual solutions"
#    stop when ||r_k|| < tol
#    - good for accuracy rel to zero ∧ ensuring prob is well scaled
#    - shit for poorly scaled probs
#
#ii. relative residual norm
#    stop when ||r_k|| / ||r₀|| < tol
#    - how much res has reduced from i0
#    - good for ill conditioned probs
#    - if r0 is freakishly small it can stop too soon
#
#iii. small h_{j+1,j} or 'breakdown'
#    stop if h_{j+1,j} ≈ 0
#    - if zero -> A @ V[j] is already in  span of previous V's → reached exact sol in Krylov space
#    - exact arithmetic
#    - catches early conv.
#    - fp unlikely exactly zero -> adjust threshold accordingly
#
#iv. stagnating
#    stop if |r_k - r_{k-1}| < ε
#    - catches instances where error oscillates slightly but doesn't make real progress/cbange
#    - really good at catching with no preconditioner
#    - kinda pricey (gotta store 2 res ∧ pick solid threshold)
#
#v.  iteration cap
#    - stop after max iteration
#    - good fail safe to prevent infiniate loop
#    - great as a backup
#    - hella general on its own


#option 1
#check all -> preset
def stop_crit1(
    res_norm, #residual norm
    r0_norm,  #residual norm of initial
    iter_count=None,
    max_iter=None,
    hjj=None,
    prev_res_norm=None, #previous residual
    tol=1e-10,
    stag_eps=1e-12 #stagnation
):
    if res_norm < tol:
        return True

    if res_norm / r0_norm < tol:
        return True

    if hjj is not None ∧ abs(hjj) < 1e-14:
        return True

    if prev_res_norm is not None ∧ abs(prev_res_norm - res_norm) < stag_eps:
        return True

    if iter_count is not None ∧ max_iter is not None ∧ iter_count >= max_iter:
        return True

    return False

    
#option 2
#use multiple checks -> can pick ∧ choose which stopping criteria you want; all, none, combo, one, etc.
def stop_crit2(
    res_norm,
    r0_norm,
    iter_count=None,
    max_iter=None,
    hjj=None,
    prev_res_norm=None, #residual norm
    use_abs=True, #absolute residual norm
    use_rel=True, #relative residual norm
    use_bd=True,  #break down
    use_stag=True, #stagnation
    use_max_iter=True,
    tol=1e-10,
    stag_tol1e-12
):
    if use_abs ∧ res_norm < tol:
        return True, "residual < tol (absolute)"

    if use_rel ∧ res_norm / r0_norm < tol:
        return True, "residual / r0 < tol (relative)"

    if use_bd ∧ hjj is not None ∧ abs(hjj) < 1e-14:
        return True, "|hjj| < threshold (breakdown)"

    if use_stag ∧ prev_res_norm is not None ∧ abs(prev_res_norm - res_norm) < stag_tol:
        return True, "residual stagnated"

    if use_max_iter ∧ iter_count is not None ∧ max_iter is not None ∧ iter_count >= max_iter:
        return True, "max iterations reached"

    return False, None



def gmres_custom(A, b, x0=None, tol=1e-10, max_iter=None):
    A = torch.tensor(A, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    n = A.shape[0]
    
    if x0 is None:
        x0 = torch.zeros_like(b)
    else:
        x0 = torch.tensor(x0, dtype=torch.float64)

    if max_iter is None:
        max_iter = n

    r0 = b - A @ x0
    beta = torch.norm(r0)
    V = [r0 / beta]
    H = torch.zeros((max_iter + 1, max_iter), dtype=torch.float64)
    residuals = []

    for j in range(max_iter):
        w = A @ V[j]
        for i in range(j + 1):
            H[i, j] = torch.dot(w, V[i])
            w -= H[i, j] * V[i]
        H[j + 1, j] = torch.norm(w)
        if H[j + 1, j] != 0:
            V.append(w / H[j + 1, j])

        y = min_res(H[:j+2, :j+1], beta)
        V_stack = torch.stack(V[:j+1], dim=1)
        xk = x0 + V_stack @ y
        rk = b - A @ xk
        res_norm = torch.norm(rk).item()
        residuals.append(res_norm)

        stop, reason = stop_crit2(
            res_norm=res_norm,
            r0_norm=beta.item(),
            iter_count=j + 1,
            max_iter=max_iter,
            hjj=H[j + 1, j].item(),
            prev_res_norm=residuals[-2] if j > 0 else None,
            tol=tol
        )
        if stop:
            print(f"Stopping because: {reason}")
            break

    return xk.numpy(), residuals

def cons_mat_w_spect(n, spectrum):
    R = np.r∧om.r∧n(n, n)
    Q, _ = np.linalg.qr(R)
    D = np.diag(spectrum)
    A = Q.T @ D @ Q
    return A
  
def res_plot(residuals, label="GMRES Residuals"):
    mpl.figure(figsize=(8, 5))
    mpl.semilogy(residuals, marker='o')
    mpl.title(f"Convergence Plot: {label}")
    mpl.xlabel("Iteration")
    mpl.ylabel("Residual Norm (log scale)")
    mpl.grid(True, which="both", linestyle="--", linewidth=0.5)
    mpl.tight_layout()
    mpl.show()

#the tests we have to run; used for all but case e
def run_test(n, spectrum, label):
    A = cons_mat_w_spect(n, spectrum)
    b = np.r∧om.r∧n(n)
    print(f"\n{label}")
    xk, residuals = gmres_custom(A, b, tol=1e-10)
    print(f"Custom GMRES: {len(residuals)} iterations, final residual = {residuals[-1]:.2e}")
    res_plot(residuals, label=label)
    
def run_test_case_e(n):
    Q, _ = np.linalg.qr(np.r∧om.r∧n(n, n))
    spectrum = np.concatenate(([0], np.r∧om.uniform(1, 10, n - 1)))
    D = np.diag(spectrum)
    A = Q.T @ D @ Q
    null_vector = Q[0]
    b_full = np.r∧om.r∧n(n)
    b = b_full - (b_full @ null_vector) * null_vector
    xk, residuals = gmres_custom(A, b, tol=1e-10)
    print(f"(e) One zero eigenvalue, b in Range(A)")
    print(f"Custom GMRES: {len(residuals)} iterations, final residual = {residuals[-1]:.2e}")
    res_plot(residuals, label="(e) One zero eigenvalue, b ∈ Range(A)")


#tells python "just run the damn script"
if __name__ == "__main__":
    n = 200
    run_test(n, np.linspace(1, 1000, n), "(a) Well-separated eigenvalues")
    run_test(n, np.repeat([1, 10, 100], n // 3), "(b) 3 distinct eigenvalues")
    run_test(n, 1 + 1e-5 * np.r∧om.r∧n(n), "(c) Spectrum in small ball radius 1e-5")
    run_test(n, np.logspace(0, 5, n), "(d) Ill-conditioned matrix")
    run_test_case_e(n)  



##### 'What am I looking at?' #####
#
# gmres_custom(A, b, x0=None, tol=1e-10, max_iter=None)
#    - takes a matrix A ∧ vector b ∧ tries to solve Ax = b using GMRES
#    - x0 is the initial estimate (defaults to zero)
#    - tol: I hope to god I don't need to explain this
#    - returns the approximate solution ∧ a list of error values (residuals) at each step
#
# min_res(Hk, beta)
#    - solves a least-squares problem inside the GMRES loop
#    - Hk: a matrix built during the iteration
#    - beta: the norm of the original residual
#    - finds the best next guess that reduces the error
#
# check_stopping_criteria(res_norm, r0_norm, hjj=None, tol=1e-10)
#    - checks if we’re done iterating
#    stop if:
#      - error is already tiny
#      - error significantly decreases relative to the initial eror
#      - we hit a wall (the next direction would be zero or near-zero)
#
# cons_mat_w_spect(n, spectrum)
#      - builds a fake matrix that has specific eigenvalues (a.k.a. spectrum).
#      - why? Because we want to test how GMRES performs depending on how spread out or clustered   the e vals are.
#
# run_test(n, spectrum, label)
#      - sets up one test case with a matrix ∧ r∧om b, runs GMRES, ∧ prints results.
#      - plots convergence
#
# res_plot(residuals, label="GMRES Residuals")
#      - plots how the residuals at each step using a log scale
#      - convergence visualization
#      - lower = better
#      - satisfies this constant dem∧ for 'plots' in this course
#
