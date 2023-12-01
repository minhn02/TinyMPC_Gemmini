#include <iostream>
#include "gemmini.h"

#include "admm.hpp"
#include "tinympc/glob_opts.hpp"

#include <Eigen/Dense>
#include <vector>
#include <cstdlib>

#define DEBUG_MODULE "TINYALG"

using namespace Eigen;

extern "C"
{
    static uint64_t startTimestamp;

    static uint64_t read_cycles() {
        uint64_t cycles;
        asm volatile ("rdcycle %0" : "=r" (cycles));
        return cycles;
    }

    void tiled_matmul_auto_eigen (
        const Matrix<float, Dynamic, Dynamic, RowMajor>&A,
        const Matrix<float, Dynamic, Dynamic, RowMajor>&B,
        Matrix<float, Dynamic, Dynamic, RowMajor>&C,
        bool transpose_A, bool transpose_B) 
    {
            int i = transpose_A ? A.cols() : A.rows();
            int j = transpose_B ? B.rows() : B.cols();
            int k = transpose_B ? B.cols() : B.rows();
            tiled_matmul_auto(i, j, k,
                    A.data(), B.data(), NULL, C.data(),
                    transpose_A ? i : k, transpose_B ? k : j, j, j,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
                    transpose_A, transpose_B,
                    false, false,
                    0,
                    WS
                    );
    }

    /**
     * Do backward Riccati pass then forward roll out
     */
    void update_primal(TinySolver *solver)
    {
        backward_pass_grad(solver);
        forward_pass(solver);
    }

    /**
     * Update linear terms from Riccati backward pass
     */
    void backward_pass_grad(TinySolver *solver)
    {
        Matrix<float, Dynamic, Dynamic, RowMajor> B_p(NINPUTS, 1);
        Matrix<float, Dynamic, Dynamic, RowMajor> dcol(NINPUTS, 1);
        Matrix<float, Dynamic, Dynamic, RowMajor> K_r(NSTATES, 1);
        Matrix<float, Dynamic, Dynamic, RowMajor> AmBKt_p(NSTATES, 1);

        for (int i = NHORIZON - 2; i >= 0; i--)
        {
            // (solver->work->d.col(i)).noalias() = solver->cache->Quu_inv * (solver->work->Bdyn.transpose() * solver->work->p.col(i + 1) + solver->work->r.col(i));
            tiled_matmul_auto_eigen(solver->work->Bdyn, solver->work->p.col(i + 1), B_p, true, false);
            tiled_matmul_auto_eigen(solver->cache->Quu_inv, B_p + solver->work->r.col(i), dcol, true, false);
            (solver->work->d.col(i)).noalias() = dcol;

            // (solver->work->p.col(i)).noalias() = solver->work->q.col(i) + solver->cache->AmBKt.lazyProduct(solver->work->p.col(i + 1)) - (solver->cache->Kinf.transpose()).lazyProduct(solver->work->r.col(i)); // + solver->cache->coeff_d2p * solver->work->d.col(i); // coeff_d2p always appears to be zeros (faster to comment out)
            tiled_matmul_auto_eigen(solver->cache->Kinf, solver->work->r.col(i), K_r, true, false);
            tiled_matmul_auto_eigen(solver->cache->AmBKt, solver->work->p.col(i + 1), AmBKt_p, false, false);
            (solver->work->p.col(i)).noalias() = solver->work->q.col(i) + AmBKt_p - K_r;
        }
    }

    /**
     * Use LQR feedback policy to roll out trajectory
     */
    void forward_pass(TinySolver *solver)
    {
        Matrix<float, Dynamic, Dynamic, RowMajor> Kinf_x(NINPUTS, 1);
        Matrix<float, Dynamic, Dynamic, RowMajor> A_x(NSTATES, 1);
        Matrix<float, Dynamic, Dynamic, RowMajor> B_u(NSTATES, 1);

        for (int i = 0; i < NHORIZON - 1; i++)
        {
            tiled_matmul_auto_eigen(solver->cache->Kinf, solver->work->x.col(i), Kinf_x, false, false);
            (solver->work->u.col(i)).noalias() = -Kinf_x - solver->work->d.col(i);
            // solver->work->u.col(i) << .001, .02, .3, 4;
            // DEBUG_PRINT("u(0): %f\n", solver->work->u.col(0)(0));
            // multAdyn(solver->Ax->cache.Adyn, solver->work->x.col(i));

            tiled_matmul_auto_eigen(solver->work->Adyn, solver->work->x.col(i), A_x, false, false);
            tiled_matmul_auto_eigen(solver->work->Bdyn, solver->work->u.col(i), B_u, false, false);
            (solver->work->x.col(i + 1)).noalias() = A_x + B_u;
        }
    }

    /**
     * Project slack (auxiliary) variables into their feasible domain, defined by
     * projection functions related to each constraint
     * TODO: pass in meta information with each constraint assigning it to a
     * projection function
     */
    void update_slack(TinySolver *solver)
    {
        solver->work->znew = solver->work->u + solver->work->y;
        solver->work->vnew = solver->work->x + solver->work->g;

        // Box constraints on input
        if (solver->settings->en_input_bound)
        {
            solver->work->znew = solver->work->u_max.cwiseMin(solver->work->u_min.cwiseMax(solver->work->znew));
        }

        // Box constraints on state
        if (solver->settings->en_state_bound)
        {
            solver->work->vnew = solver->work->x_max.cwiseMin(solver->work->x_min.cwiseMax(solver->work->vnew));
        }
    }

    /**
     * Update next iteration of dual variables by performing the augmented
     * lagrangian multiplier update
     */
    void update_dual(TinySolver *solver)
    {
        solver->work->y = solver->work->y + solver->work->u - solver->work->znew;
        solver->work->g = solver->work->g + solver->work->x - solver->work->vnew;
    }

    /**
     * Update linear control cost terms in the Riccati feedback using the changing
     * slack and dual variables from ADMM
     */
    void update_linear_cost(TinySolver *solver)
    {
        // solver->work->r = -(solver->Uref.array().colwise() * solver->work->r.array()); // Uref = 0 so commented out for speed up. Need to uncomment if using Uref

        Matrix<float, Dynamic, Dynamic, RowMajor> Xref_Pinf(NSTATES, 1);

        solver->work->r = -solver->cache->rho * (solver->work->znew - solver->work->y);
        // TODO does Gemmini do component-wise multiplication?
        solver->work->q = -(solver->work->Xref.array().colwise() * solver->work->Q.array());
        (solver->work->q).noalias() -= solver->cache->rho * (solver->work->vnew - solver->work->g);
        // solver->work->p.col(NHORIZON - 1) = -(solver->work->Xref.col(NHORIZON - 1).transpose().lazyProduct(solver->cache->Pinf));
        tiled_matmul_auto_eigen(solver->work->Xref.col(NHORIZON - 1), solver->cache->Pinf, Xref_Pinf, true, false);
        solver->work->p.col(NHORIZON - 1) = -Xref_Pinf;
        solver->work->p.col(NHORIZON - 1) -= solver->cache->rho * (solver->work->vnew.col(NHORIZON - 1) - solver->work->g.col(NHORIZON - 1));
    }

    
    int tiny_solve(TinySolver *solver)
    {
        // Initialize variables
        solver->work->status = 11;  // TINY_UNSOLVED
        solver->work->iter = 1;

        forward_pass(solver);
        update_slack(solver);
        update_dual(solver);
        update_linear_cost(solver);
        for (int i = 0; i < solver->settings->max_iter; i++)
        {

            // Solve linear system with Riccati and roll out to get new trajectory
            update_primal(solver);

            // Project slack variables into feasible domain
            update_slack(solver);

            // Compute next iteration of dual variables
            update_dual(solver);

            // Update linear control cost terms using reference trajectory, duals, and slack variables
            update_linear_cost(solver);

            if (solver->work->iter % solver->settings->check_termination == 0)
            {
                solver->work->primal_residual_state = (solver->work->x - solver->work->vnew).cwiseAbs().maxCoeff();
                solver->work->dual_residual_state = ((solver->work->v - solver->work->vnew).cwiseAbs().maxCoeff()) * solver->cache->rho;
                solver->work->primal_residual_input = (solver->work->u - solver->work->znew).cwiseAbs().maxCoeff();
                solver->work->dual_residual_input = ((solver->work->z - solver->work->znew).cwiseAbs().maxCoeff()) * solver->cache->rho;

                if (solver->work->primal_residual_state < solver->settings->abs_pri_tol &&
                    solver->work->primal_residual_input < solver->settings->abs_pri_tol &&
                    solver->work->dual_residual_state < solver->settings->abs_dua_tol &&
                    solver->work->dual_residual_input < solver->settings->abs_dua_tol)
                {
                    solver->work->status = 1;  // TINY_SOLVED
                    return 0; // 0 means solved with no error
                }
            }

            // Save previous slack variables
            solver->work->v = solver->work->vnew;
            solver->work->z = solver->work->znew;

            solver->work->iter += 1;

            // std::cout << solver->work->primal_residual_state << std::endl;
            // std::cout << solver->work->dual_residual_state << std::endl;
            // std::cout << solver->work->primal_residual_input << std::endl;
            // std::cout << solver->work->dual_residual_input << "\n" << std::endl;
        }
        return 1;
    }

} /* extern "C" */