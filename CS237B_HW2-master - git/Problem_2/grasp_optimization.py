#!/usr/bin/env python

import cvxpy as cp
import numpy as np
import pdb  

from utils import *

def solve_socp(x, As, cs, F, g, h, verbose=False):
    """
    Solves an SOCP of the form:

    minimize(h^T x)
    subject to:
        ||A_i x + b_i||_2 <= c_i^T x + d_i    for all i
        F x == g

    Args:
        x       - cvx variable.
        As      - list of A_i numpy matrices.
        bs      - list of b_i numpy vectors.
        cs      - list of c_i numpy vectors.
        ds      - list of d_i numpy vectors.
        F       - numpy matrix.
        g       - numpy vector.
        h       - numpy vector.
        verbose - whether to print verbose cvx output.

    Return:
        x - the optimal value as a numpy array, or None if the problem is
            infeasible or unbounded.
    """
    objective = cp.Minimize(h.T @ x)
    constraints = []
    for A, c in zip(As, cs):
        constraints.append(cp.SOC(c.T @ x, A @ x))
    constraints.append(F @ x == g)
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)

    print("prob status", prob.status)

    if prob.status in ['infeasible', 'unbounded']:
        return None

    return x.value

def grasp_optimization(grasp_normals, points, friction_coeffs, wrench_ext):
    """
    Solve the grasp force optimization problem as an SOCP. Handles 2D and 3D cases.

    Args:
        grasp_normals   - list of M surface normals at the contact points, pointing inwards.
        points          - list of M grasp points p^(i).
        friction_coeffs - friction coefficients mu_i at each point p^(i).
        wrench_ext      - external wrench applied to the object.

    Return:
        f
    """
    D = points[0].shape[0]  # planar: 2, spatial: 3
    N = wrench_size(D)      # planar: 3, spatial: 6
    M = len(points)
    transformations = [compute_local_transformation(n) for n in grasp_normals]

    ########## Your code starts here ##########
    As = []
    cs = []

    n = 0
    for i in range(M): n += len(grasp_normals[i])
    x = cp.Variable(n + 1)

    F = np.zeros((N,1))
    i = 0
    for point, mu, T in zip(points, friction_coeffs,transformations):
        # A
        A = np.zeros((D-1,n +1))
        A[:,D*i:(D*i)+(D-1)] = np.identity(D)[0:D-1,0:D-1]
        As.append(A)
        A2 = np.zeros((D,n +1))
        A2[:,D*i:D*i + D] = np.identity(D)
        As.append(A2)

        # c
        c = np.zeros(n + 1)
        c[D*i+(D-1)] = mu
        cs.append(c)
        c2 = np.zeros(n+1)
        c2[-1] = 1
        cs.append(c2)

        # F
        P = cross_matrix(point)
        tpt = np.vstack([T,P@T])
        F = np.hstack((F,tpt))
        i+=1

    F = np.hstack((F[:,1:],np.zeros((N,1))))

    #g
    g = np.array(-1*wrench_ext)

    #h
    h = np.zeros(n)
    h = np.append(h,1)
    
    x = solve_socp(x, As, cs, F, g, h, verbose=False)

    print("x",x)

    # TODO: extract the grasp forces from x as a stacked 1D vector
    f = x[:-1]

    ########## Your code ends here ##########

    # Transform the forces to the global frame
    F = f.reshape(M,D)
    forces = [T.dot(f) for T, f in zip(transformations, F)]


    return forces

def precompute_force_closure(grasp_normals, points, friction_coeffs):
    """
    Precompute the force optimization problem so that force closure grasps can
    be found for any arbitrary external wrench without redoing the optimization.

    Args:
        grasp_normals   - list of M contact normals, pointing inwards from the object surface.
        points          - list of M contact points p^(i).
        friction_coeffs - friction coefficients mu_i at each point p^(i).

    Return:
        force_closure(wrench_ext) - a function that takes as input an external wrench and
                                    returns a set of forces that maintains force closure.
    """
    D = points[0].shape[0]  # planar: 2, spatial: 3
    N = wrench_size(D)      # planar: 3, spatial: 6
    M = len(points)

    print("pre compute force")

    ########## Your code starts here ##########
    # TODO: Precompute the optimal forces for the 12 signed unit external
    #       wrenches and store them as rows in the matrix F. This matrix will be
    #       captured by the returned force_closure() function.
    F = np.zeros((2*N, M*D))
    for i in range(M):
        w1 = np.zeros(N)
        w2 = np.zeros(N)
        w1[i] = 1
        w2[i] = -1
        f1 = grasp_optimization(grasp_normals, points, friction_coeffs, w1)
        f2 = grasp_optimization(grasp_normals, points, friction_coeffs, w2)
        F[2*i:2*i + 2,:] = np.vstack((np.hstack(f1),np.hstack(f2)))


    ########## Your code ends here ##########

    def force_closure(wrench_ext):
        """
        Return a set of forces that maintain force closure for the given
        external wrench using the precomputed parameters.

        Args:
            wrench_ext - external wrench applied to the object.

        Return:
            f - grasp forces as a list of M numpy arrays.
        """

        ########## Your code starts here ##########
        # TODO: Compute the force closure forces as a stacked vector of shape (M*D)
        print("force closureee")
        f = np.zeros(M*D)

        wp = np.maximum(0,wrench_ext)
        wn = np.maximum(0,-wrench_ext)

        N = wrench_size(D)
        ww = []
        for i in range(N): 
            ww.append(wp[i])
            ww.append(wn[i])
        wf = np.hstack(ww)
        f = np.dot(wf, F)

        ########## Your code ends here ##########

        forces = [f_i for f_i in f.reshape(M,D)]
        return forces

    return force_closure
