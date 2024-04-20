/*This is a program to add a noise to a DOF in phg
Assume we have a DOF u, then we add a noise
$$
(1+\sigma rand)*u
$$
rand gives uniformly distributed random numbers in [−1, 1], and σ is a noise
level parameter.
*/
#include "forward.h"
#include "phg.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define check phgPrintf("HELLO I AM HERE LINE=%d\n", __LINE__)

FLOAT
phgQuadFaceTanBasDotTanBas(ELEMENT *e, int face, DOF *u, int M, DOF *v, int N,
                           int order) {
  GRID *g = u->g;
  int i, j, k, nvalues, v0, v1, v2;
  FLOAT d, d0, lambda[Dim + 1];
  FLOAT *buffer;
  const FLOAT *bas0, *bas, *p, *w;
  QUAD *quad;
  const FLOAT n[3]; /* out normal vector */
  FLOAT u_cross_n[3], v_cross_n[3];
  DOF_TYPE *type_u, *type_v;

  type_u = (DofIsHP(u) ? u->hp->info->types[u->hp->max_order] : u->type);
  type_v = (DofIsHP(v) ? v->hp->info->types[v->hp->max_order] : v->type);
  assert(DofTypeDim(u) == DofTypeDim(v));

  assert(!SpecialDofType(v->type));
  assert(face >= 0 && face <= 3);

  if (order < 0) {
    i = DofTypeOrder(u, e);
    j = DofTypeOrder(v, e);
    if (i < 0)
      i = j;
    order = i + j;
  }
  quad = phgQuadGetQuad2D(order);

  v0 = GetFaceVertex(face, 0);
  v1 = GetFaceVertex(face, 1);
  v2 = GetFaceVertex(face, 2);
  lambda[face] = 0.;

  nvalues = DofTypeDim(v);

  if (nvalues != Dim)
    phgError(1, "%s: dimensions mismatch\n", __func__);

  if (N != M || u != v)
    buffer = phgAlloc(nvalues * sizeof(*buffer));
  else
    buffer = NULL;
  // buffer = phgAlloc(nvalues * sizeof(*buffer));
  p = quad->points;
  w = quad->weights;

  d = 0.;
  phgGeomGetFaceOutNormal(g, e, face, n);

  for (i = 0; i < quad->npoints; i++) {
    lambda[v0] = *(p++);
    lambda[v1] = *(p++);
    lambda[v2] = *(p++);

    bas0 = (FLOAT *)type_u->BasFuncs(u, e, M, M + 1, lambda);
    if (N == M && u == v) {
      bas = bas0;
    } else {
      memcpy(buffer, bas0, nvalues * sizeof(*buffer));
      bas0 = buffer;
      bas = type_v->BasFuncs(v, e, N, N + 1, lambda);
    }

    // bas = (FLOAT *)type_v->BasFuncs(v, e, N, N + 1, lambda);

    u_cross_n[0] = bas0[1] * n[2] - bas0[2] * n[1];
    u_cross_n[1] = bas0[2] * n[0] - bas0[0] * n[2];
    u_cross_n[2] = bas0[0] * n[1] - bas0[1] * n[0];

    v_cross_n[0] = bas[1] * n[2] - bas[2] * n[1];
    v_cross_n[1] = bas[2] * n[0] - bas[0] * n[2];
    v_cross_n[2] = bas[0] * n[1] - bas[1] * n[0];

    d0 = 0.0;
    for (j = 0; j < nvalues; j++) {
      d0 += u_cross_n[j] * v_cross_n[j];
    }
    d += d0 * *(w++);
  }

  phgFree(buffer);

  return d * phgGeomGetFaceArea(u->g, e, face);
}

FLOAT
phgQuadFaceTanDofDotTanBas(ELEMENT *e, int face, DOF *u, DOF *v, int N,
                           int order) {
  GRID *g = u->g;
  int i, j, k, nvalues, v0, v1, v2;
  FLOAT d, d0, lambda[Dim + 1];
  FLOAT *dof, *buffer;
  const FLOAT *bas, *p, *w;
  QUAD *quad;
  const FLOAT n[3]; /* out normal vector */
  DOF_TYPE *type;
  FLOAT u_dot_n, v_dot_n;

  assert(!SpecialDofType(v->type));
  assert(face >= 0 && face <= 3);

  type = (DofIsHP(v) ? v->hp->info->types[v->hp->max_order] : v->type);
  if (order < 0) {
    i = DofTypeOrder(u, e);
    j = DofTypeOrder(v, e);
    if (i < 0)
      i = j;
    order = i + j;
  }
  quad = phgQuadGetQuad2D(order);

  v0 = GetFaceVertex(face, 0);
  v1 = GetFaceVertex(face, 1);
  v2 = GetFaceVertex(face, 2);
  lambda[face] = 0.;

  nvalues = DofTypeDim(v);

  if (nvalues != Dim)
    phgError(1, "%s: dimensions mismatch\n", __func__);

  buffer = phgAlloc(nvalues * sizeof(*buffer));
  p = quad->points;
  w = quad->weights;

  d = 0.;
  phgGeomGetFaceOutNormal(g, e, face, n);

  for (i = 0; i < quad->npoints; i++) {
    lambda[v0] = *(p++);
    lambda[v1] = *(p++);
    lambda[v2] = *(p++);

    dof = phgDofEval(u, e, lambda, buffer);
    bas = (FLOAT *)type->BasFuncs(v, e, N, N + 1, lambda);
    u_dot_n = n[0] * dof[0] + n[1] * dof[1] + n[2] * dof[2];
    v_dot_n = n[0] * bas[0] + n[1] * bas[1] + n[2] * bas[2];

    d0 = 0.0;
    for (j = 0; j < nvalues; j++) {
      d0 += *(bas++) * *(dof++);
    }
    d0 -= u_dot_n * v_dot_n;
    d += d0 * *(w++);
  }

  phgFree(buffer);

  return d * phgGeomGetFaceArea(u->g, e, face);
}

static void build_forward_linear_system(SOLVER *solver, DOF *E_h_re,
                                        DOF *E_h_im, DOF *f_re, DOF *f_im,
                                        DOF *epsilon, DOF *epsilon_im) {
  int N = E_h_re->type->nbas; /* number of basis functions in an element */
  int M = 2 * N;
  GRID *g = E_h_re->g;
  ELEMENT *e;
  int i, j, ii, jj, face, i1, j1;
  FLOAT A[M][M], B[M];
  INT I[M];
  FLOAT stiffness, mass;
  FLOAT tmp_re, tmp_im, tmp1_re, tmp1_im, mat;

  assert(E_h_re->dim == 1);
  // check;
  //   assert(!SpecialDofType(E_h_re->type) );
  // check;

  ForAllElements(g, e) { /* \PHGindex{ForAllElements} */
    /* compute \int (\curl\phi_j \cdot \curl\phi_i-k^2\epsilon \phi_j\phi_k)+
     * making use of symmetry */
    for (i = 0; i < N; i++) {
      I[i] = phgSolverMapE2L(solver, 0, e, i);
      I[i + N] = phgSolverMapE2L(solver, 1, e, i);

      for (j = 0; j <= i; j++) {
        stiffness =
            phgQuadCurlBasDotCurlBas(e, E_h_re, j, E_h_re, i, QUAD_DEFAULT);
        mass = phgQuadBasABas(e, E_h_re, j, epsilon, E_h_re, i, QUAD_DEFAULT);
        A[j][i] = A[i][j] = stiffness - k * k * mass;

        A[j + N][i + N] = A[i + N][j + N] = -(stiffness - k * k * mass);
        A[i][j + N] = A[i + N][j] = A[j + N][i] = A[j][i + N] =
            k * k *
            phgQuadBasABas(e, E_h_re, j, epsilon_im, E_h_re, i, QUAD_DEFAULT);
      }
    }
    // check;
    for (i = 0; i < N; i++) { /* loop on basis functions */
      // tmp_re= phgQuadDofDotBas(e, E_inc_re, E_h_re, i, QUAD_DEFAULT);
      // tmp_im=phgQuadDofDotBas(e, f_h_re, E_h_re, i, QUAD_DEFAULT);
      B[i] = k * k * phgQuadDofDotBas(e, f_re, E_h_re, i, QUAD_DEFAULT);
      //+      phgQuadDofDotBas(e, f_h_re, E_h_re, i, QUAD_DEFAULT);

      B[i + N] = -(k * k * phgQuadDofDotBas(e, f_im, E_h_re, i, QUAD_DEFAULT));
      //                  + phgQuadDofDotBas(e, f_h_im, E_h_re, i,
      //                  QUAD_DEFAULT));
      phgSolverAddMatrixEntries(solver, 1, I + i, 2 * N, I, A[i]);
      phgSolverAddMatrixEntries(solver, 1, I + i + N, 2 * N, I, A[i + N]);
    }
    phgSolverAddRHSEntries(solver, (2 * N), I, B);
    /* compute local matrix and RHS */
    // check;
    for (face = 0; face < NFace; face++) {
      if (e->bound_type[face] & BDRY_MASK) {
        int n = phgDofGetBasesOnFace(E_h_re, e, face, NULL);
        SHORT bases[n];
        phgDofGetBasesOnFace(E_h_re, e, face, bases);
        for (ii = 0; ii < n; ii++) {
          i = phgSolverMapE2L(solver, 0, e, bases[ii]);
          i1 = phgSolverMapE2L(solver, 1, e, bases[ii]);
          for (jj = 0; jj < n; jj++) {
            j = phgSolverMapE2L(solver, 0, e, bases[jj]);
            j1 = phgSolverMapE2L(solver, 1, e, bases[jj]);
            mat =
                k * phgQuadFaceTanBasDotTanBas(e, face, E_h_re, bases[jj],
                                               E_h_re, bases[ii], QUAD_DEFAULT);
            phgSolverAddMatrixEntry(solver, i, j1, mat);
            // phgSolverAddMatrixEntry(solver, i1, j, (-1. * mat));
            phgSolverAddMatrixEntry(solver, i1, j, (mat));
          }
          //  tmp_re = phgQuadFaceDofDotBas(e, face, z_h_re, DOF_PROJ_CROSS,
          //  E_h_re,
          //                            bases[ii], QUAD_DEFAULT);
          // tmp_im = phgQuadFaceDofDotBas(e, face, z_h_im, DOF_PROJ_CROSS,
          // E_h_re,
          //                           bases[ii], QUAD_DEFAULT);
          // tmp1_re = phgQuadFaceTanDofDotTanBas(e, face, w_h_re, E_h_re,
          //                                    bases[ii], QUAD_DEFAULT);
          // tmp1_im = phgQuadFaceTanDofDotTanBas(e, face, w_h_im, E_h_re,
          //                                    bases[ii], QUAD_DEFAULT);
          // printf("%d = %lf , %lf\n",i,tmp_re + k * tmp1_im,tmp_im - k *
          // tmp1_re);
          // phgSolverAddRHSEntry(solver, i, (tmp_re + k * tmp1_im));
          // phgSolverAddRHSEntry(solver, i1, (tmp_im - k * tmp1_re));
          // phgSolverAddRHSEntry(solver, i1, -(tmp_im - k * tmp1_re));
        }
      }
    }
  }
}
static void func_g_re(FLOAT x, FLOAT y, FLOAT z, FLOAT *value) {
  /* Re(E_inc)= -k\mathbf{p}\sin{\mathrm{i}k\mathbf{x}\cdot\mathbf{n}}*/
  FLOAT tmp, tmp_cos, tmp_sin;

  tmp = k * (x * nv[0] + y * nv[1] + z * nv[2]);
  // tmp_cos = Cos(tmp);

  tmp_sin = Sin(tmp);
  *(value++) = -k * p[0] * tmp_sin;
  *(value++) = -k * p[1] * tmp_sin;
  *(value++) = -k * p[2] * tmp_sin;
}
static void func_g_im(FLOAT x, FLOAT y, FLOAT z, FLOAT *value) {
  /* Im(E_inc)= k\mathbf{p}\cos{\mathrm{i}k\mathbf{x}\cdot\mathbf{n}}*/
  FLOAT tmp, tmp_cos, tmp_sin;
  // FLOAT p[3];
  // FLOAT n[3];
  tmp = k * (x * nv[0] + y * nv[1] + z * nv[2]);
  tmp_cos = Cos(tmp);

  // tmp_sin = Sin(tmp);
  *(value++) = k * p[0] * tmp_cos;
  *(value++) = k * p[1] * tmp_cos;
  *(value++) = k * p[2] * tmp_cos;
}
static void build_forward_linear_system_etotal(SOLVER *solver, DOF *E_h_re,
                                               DOF *E_h_im, DOF *epsilon,
                                               DOF *epsilon_im, DOF *z_h_re,
                                               DOF *z_h_im, DOF *w_h_re,
                                               DOF *w_h_im) {
  int N = E_h_re->type->nbas; /* number of basis functions in an element */
  int M = 2 * N;
  GRID *g = E_h_re->g;
  ELEMENT *e;
  int i, j, ii, jj, face, i1, j1;
  FLOAT A[M][M], B[M], buffer_re[N], buffer_im[N], rhs[M];
  INT I[M];
  FLOAT stiffness, mass;
  FLOAT tmp_re, tmp_im, tmp1_re, tmp1_im, mat;

  assert(E_h_re->dim == 1);
  // check;
  //   assert(!SpecialDofType(E_h_re->type) );
  // check;

  ForAllElements(g, e) { /* \PHGindex{ForAllElements} */
    /* compute \int (\curl\phi_j \cdot \curl\phi_i-k^2\epsilon \phi_j\phi_k)+
     * making use of symmetry */
    for (i = 0; i < N; i++) {
      I[i] = phgSolverMapE2L(solver, 0, e, i);
      I[i + N] = phgSolverMapE2L(solver, 1, e, i);

      for (j = 0; j <= i; j++) {
        stiffness =
            phgQuadCurlBasDotCurlBas(e, E_h_re, j, E_h_re, i, QUAD_DEFAULT);
        mass = phgQuadBasABas(e, E_h_re, j, epsilon, E_h_re, i, QUAD_DEFAULT);
        A[j][i] = A[i][j] = stiffness - k * k * mass;

        A[j + N][i + N] = A[i + N][j + N] = -(stiffness - k * k * mass);
        A[i][j + N] = A[i + N][j] = A[j + N][i] = A[j][i + N] =
            k * k *
            phgQuadBasABas(e, E_h_re, j, epsilon_im, E_h_re, i, QUAD_DEFAULT);
      }
    }
    // check;
    for (i = 0; i < N; i++) { /* loop on basis functions */
      if (phgDofDirichletBC(E_h_re, e, i, func_g_re, buffer_re, rhs + i,
                            DOF_PROJ_CROSS)) {
        phgDofDirichletBC(E_h_im, e, i, func_g_im, buffer_im, rhs + i + N,
                          DOF_PROJ_CROSS);
        phgSolverAddMatrixEntries(solver, 1, I + i, N, I, buffer_re);
        phgSolverAddMatrixEntries(solver, 1, I + i + N, N, I + N, buffer_im);
        phgSolverAddRHSEntry(solver, i, rhs[i]);
        // phgSolverAddRHSEntry(solver, i1, (tmp_im - k * tmp1_re));
        phgSolverAddRHSEntry(solver, i + N, rhs[i + N]);
      } else {

        phgSolverAddMatrixEntries(solver, 1, I + i, 2 * N, I, A[i]);
        phgSolverAddMatrixEntries(solver, 1, I + i + N, 2 * N, I, A[i + N]);
      }
    }

    for (face = 0; face < NFace; face++) {
      if (e->bound_type[face] & BDRY_USER1) {
        int n = phgDofGetBasesOnFace(E_h_re, e, face, NULL);
        SHORT bases[n];
        phgDofGetBasesOnFace(E_h_re, e, face, bases);
        for (ii = 0; ii < n; ii++) {
          i = phgSolverMapE2L(solver, 0, e, bases[ii]);
          i1 = phgSolverMapE2L(solver, 1, e, bases[ii]);
          for (jj = 0; jj < n; jj++) {
            j = phgSolverMapE2L(solver, 0, e, bases[jj]);
            j1 = phgSolverMapE2L(solver, 1, e, bases[jj]);
            mat =
                k * phgQuadFaceTanBasDotTanBas(e, face, E_h_re, bases[jj],
                                               E_h_re, bases[ii], QUAD_DEFAULT);
            phgSolverAddMatrixEntry(solver, i, j1, mat);
            // phgSolverAddMatrixEntry(solver, i1, j, (-1. * mat));
            phgSolverAddMatrixEntry(solver, i1, j, (mat));
          }
          tmp_re = phgQuadFaceDofDotBas(e, face, z_h_re, DOF_PROJ_CROSS, E_h_re,
                                        bases[ii], QUAD_DEFAULT);
          tmp_im = phgQuadFaceDofDotBas(e, face, z_h_im, DOF_PROJ_CROSS, E_h_re,
                                        bases[ii], QUAD_DEFAULT);
          tmp1_re = phgQuadFaceTanDofDotTanBas(e, face, w_h_re, E_h_re,
                                               bases[ii], QUAD_DEFAULT);
          tmp1_im = phgQuadFaceTanDofDotTanBas(e, face, w_h_im, E_h_re,
                                               bases[ii], QUAD_DEFAULT);
          // printf("%d = %lf , %lf\n",i,tmp_re + k * tmp1_im,tmp_im - k *
          // tmp1_re);
          phgSolverAddRHSEntry(solver, i, (tmp_re + k * tmp1_im));
          // phgSolverAddRHSEntry(solver, i1, (tmp_im - k * tmp1_re));
          phgSolverAddRHSEntry(solver, i1, -(tmp_im - k * tmp1_re));
        }
      }
    }
  }
}
void maxwell_forward_solver_etotal(DOF *E_h_re, DOF *E_h_im, DOF *E_inc_re,
                                   DOF *E_inc_im, DOF *curlE_inc_re,
                                   DOF *curlE_inc_im, DOF *epsilon_re,
                                   DOF *epsilon_im) {

  GRID *g = E_h_re->g;
  SOLVER *solver, *pc = NULL;

  phgPrintf(" \n------------------Solve the total forward Maxwell "
            "equation---------------\n");
  phgPrintf("wave number = %lf \n", k);
  phgPrintf("\n------ %" dFMT " DOF, %" dFMT " elements, mesh LIF = %g\n",
            2 * DofGetDataCountGlobal(E_h_re), g->nleaf_global, (double)g->lif);

  phgPrintf("Set up linear solver: \n");
  solver = phgSolverCreate(SOLVER_DEFAULT, E_h_re, E_h_im, NULL);
  /*
 if(k<5){
  solver = phgSolverCreate(SOLVER_DEFAULT, E_h_re, E_h_im, NULL);}
  else{
   solver = phgSolverCreate(SOLVER_SUPERLU, E_h_re, E_h_im, NULL);
  }*/
  // solver = phgSolverCreate(SOLVER_SUPERLU, E_h_re, E_h_im, NULL);
  phgPrintf("Build linear system \n");
  build_forward_linear_system_etotal(solver, E_h_re, E_h_im, epsilon_re,
                                     epsilon_im, curlE_inc_re, curlE_inc_im,
                                     E_inc_re, E_inc_im);
  phgPrintf("Solve linear system: \n");

  phgSolverSolve(solver, TRUE, E_h_re, E_h_im, NULL);
  phgPrintf("nits=%d, resid=%0.4lg \n", solver->nits, (double)solver->residual);

  phgSolverDestroy(&solver);
}
void maxwell_forward_solver(DOF *E_h_re, DOF *E_h_im, DOF *E_inc_re,
                            DOF *E_inc_im, DOF *q_re, DOF *q_im,
                            DOF *epsilon_re, DOF *epsilon_im) {

  GRID *g = E_h_re->g;
  SOLVER *solver, *pc = NULL;
  DOF *f_re, *f_im;
  f_re = phgDofNew(g, DOF_P1, 3, "f_re", DofInterpolation);
  f_im = phgDofNew(g, DOF_P1, 3, "f_im", DofInterpolation);
  check;
  phgDofMM(MAT_OP_N, MAT_OP_N, 1, 3, 1, 1.0, q_re, 0, E_inc_re, 0.0, &f_re);
  phgDofMM(MAT_OP_N, MAT_OP_N, 1, 3, 1, 1.0, q_im, 0, E_inc_re, 0.0, &f_im);
  phgDofMM(MAT_OP_N, MAT_OP_N, 1, 3, 1, -1.0, q_im, 0, E_inc_im, 1.0, &f_re);
  phgDofMM(MAT_OP_N, MAT_OP_N, 1, 3, 1, 1.0, q_re, 0, E_inc_im, 1.0, &f_im);
  check;
  phgPrintf(" \n------------------Solve the forward Maxwell "
            "equation---------------\n");
  phgPrintf("wave number = %d \n", k);
  phgPrintf("\n------ %" dFMT " DOF, %" dFMT " elements, mesh LIF = %g\n",
            2 * DofGetDataCountGlobal(E_h_re), g->nleaf_global, (double)g->lif);

  phgPrintf("Set up linear solver: \n");
  solver = phgSolverCreate(SOLVER_DEFAULT, E_h_re, E_h_im, NULL);
  /*
 if(k<5){
  solver = phgSolverCreate(SOLVER_DEFAULT, E_h_re, E_h_im, NULL);}
  else{
   solver = phgSolverCreate(SOLVER_SUPERLU, E_h_re, E_h_im, NULL);
  }*/
  // solver = phgSolverCreate(SOLVER_SUPERLU, E_h_re, E_h_im, NULL);
  phgPrintf("Build linear system \n");
  build_forward_linear_system(solver, E_h_re, E_h_im, f_re, f_im, epsilon_re,
                              epsilon_im);

  phgPrintf("Solve linear system: \n");

  phgSolverSolve(solver, TRUE, E_h_re, E_h_im, NULL);
  phgPrintf("nits=%d, resid=%0.4lg \n", solver->nits, (double)solver->residual);
  phgDofFree(&f_re);
  phgDofFree(&f_im);
  phgSolverDestroy(&solver);
}