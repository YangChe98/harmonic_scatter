/*This is a program to add a noise to a DOF in phg
Assume we have a DOF u, then we add a noise
$$
(1+\sigma rand)*u
$$
rand gives uniformly distributed random numbers in [−1, 1], and σ is a noise
level parameter.
*/
#include "phg.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define check phgPrintf("HELLO I AM HERE LINE=%d\n", __LINE__)

FLOAT sigma = 0.02;

FLOAT k;
FLOAT p[3];  // polarization
FLOAT nv[3]; // propagation direction

static int bc_map(int bctype) // boundary condition
{
  switch (bctype) {
  case 1:
    return DIRICHLET; /* Dirichlet */
  case 2:
    return BDRY_USER1; /* Neumann */
  }
}
static void func_E_inc_re(FLOAT x, FLOAT y, FLOAT z, FLOAT *value) {
  /* Re(E_inc)= -k\mathbf{p}\sin{\mathrm{i}k\mathbf{x}\cdot\mathbf{n}}*/
  FLOAT tmp, tmp_cos, tmp_sin;

  tmp = k * (x * nv[0] + y * nv[1] + z * nv[2]);
  // tmp_cos = Cos(tmp);

  tmp_sin = Sin(tmp);
  *(value++) = -k * p[0] * tmp_sin;
  *(value++) = -k * p[1] * tmp_sin;
  *(value++) = -k * p[2] * tmp_sin;
}
static void func_E_inc_im(FLOAT x, FLOAT y, FLOAT z, FLOAT *value) {
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
static void func_curlE_inc_re(FLOAT x, FLOAT y, FLOAT z, FLOAT *value) {
  /* Re(E_inc)= -k\mathbf{p}\sin{\mathrm{i}k\mathbf{x}\cdot\mathbf{n}}*/
  FLOAT tmp, tmp_cos, tmp_sin;

  tmp = k * (x * nv[0] + y * nv[1] + z * nv[2]);
  tmp_cos = Cos(tmp);

  // tmp_sin = Sin(tmp);
  *(value++) = k * k * (p[1] * nv[2] - p[2] * nv[1]) * tmp_cos;
  *(value++) = k * k * (p[2] * nv[0] - p[0] * nv[2]) * tmp_cos;
  *(value++) = k * k * (p[0] * nv[1] - p[1] * nv[0]) * tmp_cos;
}
static void func_curlE_inc_im(FLOAT x, FLOAT y, FLOAT z, FLOAT *value) {
  /* Im(E_inc)= k\mathbf{p}\cos{\mathrm{i}k\mathbf{x}\cdot\mathbf{n}}*/
  FLOAT tmp, tmp_cos, tmp_sin;
  // FLOAT p[3];
  // FLOAT n[3];
  tmp = k * (x * nv[0] + y * nv[1] + z * nv[2]);
  // tmp_cos = Cos(tmp);

  tmp_sin = Sin(tmp);
  *(value++) = k * k * (p[1] * nv[2] - p[2] * nv[1]) * tmp_sin;
  *(value++) = k * k * (p[2] * nv[0] - p[0] * nv[2]) * tmp_sin;
  *(value++) = k * k * (p[0] * nv[1] - p[1] * nv[0]) * tmp_sin;
}
static void func_epsilon_re(FLOAT x, FLOAT y, FLOAT z, FLOAT *value) {

  *(value++) = 1.;
}
static void func_epsilon_im(FLOAT x, FLOAT y, FLOAT z, FLOAT *value) {

  *(value++) = 0.;
}
static void checkBdry(GRID *g) {
  ELEMENT *e;
  int s;
  double a[5];

  a[0] = a[1] = a[2] = a[3] = a[4] = 0.;
  ForAllElements(g, e) {
    for (s = 0; s < NFace; s++) {
      FLOAT area = phgGeomGetFaceArea(g, e, s);
      if (e->bound_type[s] & BDRY_MASK) {
        if (e->bound_type[s] & DIRICHLET)
          a[0] += area;
        if (e->bound_type[s] & NEUMANN)
          a[1] += area;
        if (e->bound_type[s] & BDRY_USER1)
          a[2] += area;
        if (e->bound_type[s] & BDRY_USER2)
          a[3] += area;
      } else {
        a[4] += area;
      }
    }
  }

#if USE_MPI
  {
    double b[5];
    MPI_Reduce(a, b, 5, MPI_DOUBLE, MPI_SUM, 0, g->comm);
    memcpy(a, b, sizeof(b));
  }
#endif

  phgPrintf("Boundary types check:\n");
  phgPrintf("    Dirichlet: %g, Neumann: %g, user1: %g, user2: %g, "
            "other: %g\n",
            a[0], a[1], a[2], a[3], a[4]);

  return;
}
static FLOAT L2_Error(DOF *u, DOF *u_h) {
  ELEMENT *e;
  FLOAT tmp, result;

  tmp = 0.0;
  ForAllElements(u->g, e) {
    tmp += phgQuadDofDotDof(e, u_h, u_h, 5);
    tmp -= 2.0 * phgQuadDofDotDof(e, u_h, u, 5);
    tmp += phgQuadDofDotDof(e, u, u, 5);
  }
#if USE_MPI
  MPI_Allreduce(&tmp, &result, 1, PHG_MPI_FLOAT, MPI_SUM, u->g->comm);
#else
  result = tmp;
#endif
  return Sqrt(result);
}
static FLOAT L2_Norm(DOF *u) {
  ELEMENT *e;
  FLOAT tmp, result;

  tmp = 0.0;
  ForAllElements(u->g, e) { tmp += phgQuadDofDotDof(e, u, u, 5); }
#if USE_MPI
  MPI_Allreduce(&tmp, &result, 1, PHG_MPI_FLOAT, MPI_SUM, u->g->comm);
#else
  result = tmp;
#endif
  return Sqrt(result);
}
static void output_efieldxyz(DOF *u_h, FLOAT z, int N, INT g_flag111) {
  GRID *g = u_h->g;
  int Nnum = (N + 1);
  COORD Coord[Nnum];
  FLOAT *coord;
  FLOAT Ndofvalue[1][3 * Nnum];
  FLOAT valuex[Nnum], valuey[Nnum], valuez[Nnum];
  coord = (double *)malloc(3 * sizeof(FLOAT));

  int i, j;
  for (i = 0; i < N + 1; i++) {
    Coord[i][0] = -1. + i * 2. / N;
    Coord[i][1] = -1. + i * 2. / N;
    Coord[i][2] = -1. + i * 2. / N;
  }
  for (i = 0; i < 3 * Nnum; i++) {
    Ndofvalue[0][i] = 0.;
  }
  DOF *u_ht;
  u_ht = phgDofCopy(u_h, NULL, DOF_P1, NULL);
  phgInterGridDofEval(u_ht, Nnum, Coord, Ndofvalue[0], -1);
  phgDofFree(&u_ht);
  // u_ht = phgDofCopy(u_h1,  NULL, DOF_P1, NULL);
  // phgInterGridDofEval(u_ht,  Nnum, Coord, Ndofvalue1[0], -1);
  // phgDofFree(&u_ht);

  for (i = 0; i < Nnum; i++) {
    // valuex[i] =
    // Sqrt(Ndofvalue[0][3*i]*Ndofvalue[0][3*i]+Ndofvalue1[0][3*i]*Ndofvalue1[0][3*i]);
    // valuey[i] =
    // Sqrt(Ndofvalue[0][3*i+1]*Ndofvalue[0][3*i+1]+Ndofvalue1[0][3*i+1]*Ndofvalue1[0][3*i+1]);
    // valuez[i] =
    // Sqrt(Ndofvalue[0][3*i+2]*Ndofvalue[0][3*i+2]+Ndofvalue1[0][3*i+2]*Ndofvalue1[0][3*i+2]);
    valuex[i] = (Ndofvalue[0][3 * i]);
    valuey[i] = (Ndofvalue[0][3 * i + 1]);
    valuez[i] = (Ndofvalue[0][3 * i + 2]);
  }

  if (phgRank == 0) {

    char *sn1, *sn2, *sn3;
    sn1 = (char *)phgAlloc(30 * sizeof(char));
    sn2 = (char *)phgAlloc(30 * sizeof(char));
    sn3 = (char *)phgAlloc(30 * sizeof(char));
    FILE *fp[3];

    phgPrintf("\n\noutput Efield \n");
    *sn1 = '\0';
    *sn2 = '\0';
    *sn3 = '\0';
    sprintf(sn1, "efield/efieldEcutxyz_%4.4d", g_flag111);
    // sprintf(sn2, "efield/efieldEycutx_%4.4d", g_flag111);
    // sprintf(sn3, "efield/efieldEzcutx_%4.4d", g_flag111);
    fp[0] = fopen(sn1, "w");
    for (i = 0; i < (N + 1); i++) {
      fprintf(fp[0], "%10.9f  %10.9f  %10.9f\n", (double)valuex[i],
              (double)valuey[i], (double)valuez[i]);
    }

    // for(i = 0; i < 3; i++)  {
    //     fclose(fp[i]);
    // }
    fclose(fp[0]);
  }
}
static void output_efieldx(DOF *u_h, FLOAT z, int N, INT g_flag111) {
  GRID *g = u_h->g;
  int Nnum = (N + 1) * (N + 1);
  int Nnum1 = (N / 2) * (N + 1);
  int Nnum2 = (N + 1 - N / 2) * (N + 1);
  COORD Coord[Nnum];
  FLOAT *coord;
  FLOAT Ndofvalue[1][3 * Nnum];
  FLOAT valuex[Nnum], valuey[Nnum], valuez[Nnum];
  coord = (double *)malloc(3 * sizeof(FLOAT));

  int i, j;
  for (i = 0; i < N + 1; i++) {
    for (j = 0; j < N + 1; j++) {
      Coord[i * (N + 1) + j][0] = 0.25;
      Coord[i * (N + 1) + j][1] = -1. + i * 2. / N;
      Coord[i * (N + 1) + j][2] = -1. + j * 2. / N;
    }
  }
  for (i = 0; i < 3 * Nnum; i++) {
    Ndofvalue[0][i] = 0.;
  }
  DOF *u_ht;
  u_ht = phgDofCopy(u_h, NULL, DOF_P1, NULL);
  phgInterGridDofEval(u_ht, Nnum, Coord, Ndofvalue[0], -1);
  phgDofFree(&u_ht);

  for (i = 0; i < Nnum; i++) {
    valuex[i] = (Ndofvalue[0][3 * i]);
    valuey[i] = (Ndofvalue[0][3 * i + 1]);
    valuez[i] = (Ndofvalue[0][3 * i + 2]);
  }

  if (phgRank == 0) {

    char *sn1, *sn2, *sn3;
    sn1 = (char *)phgAlloc(30 * sizeof(char));
    sn2 = (char *)phgAlloc(30 * sizeof(char));
    sn3 = (char *)phgAlloc(30 * sizeof(char));
    FILE *fp[3];

    phgPrintf("\n\noutput Efield, flag111 = %d \n", g_flag111);
    *sn1 = '\0';
    *sn2 = '\0';
    *sn3 = '\0';
    sprintf(sn1, "efield/efieldEcutx_%4.4d", g_flag111);
    // sprintf(sn2, "efield/efieldEycutx_%4.4d", g_flag111);
    // sprintf(sn3, "efield/efieldEzcutx_%4.4d", g_flag111);
    fp[0] = fopen(sn1, "w");
    for (i = 0; i < (N + 1); i++) {
      for (j = 0; j < (N + 1); j++) {
        fprintf(
            fp[0], "%10.9f  %10.9f  %10.9f\n", (double)valuex[i * (N + 1) + j],
            (double)valuey[i * (N + 1) + j], (double)valuez[i * (N + 1) + j]);
        // fprintf(fp[0],"%10.9f \n",(double)valuex[i*(N+1)+j]);
        // fprintf(fp[1],"%10.9f \n",(double)valuey[i*(N+1)+j]);
        // fprintf(fp[2],"%10.9f \n",(double)valuez[i*(N+1)+j]);
      }
    }

    // for(i = 0; i < 3; i++)  {
    //     fclose(fp[i]);
    // }
    fclose(fp[0]);
  }
}
int main(int argc, char *argv[]) {

  INT m = 20; // the number of incient wave
  FLOAT theta =
      M_PI / 3.; // 0.;                          // incident latitudinal angel
  FLOAT phi = (2 - 1) * (M_PI) / ((FLOAT)m); // incident longitudinal angel
  FLOAT sinphi = Sin(phi);
  FLOAT cosphi = Cos(phi);
  FLOAT sintheta = Sin(theta);
  FLOAT costheta = Cos(theta);
  nv[0] = sintheta * cosphi;
  nv[1] = sintheta * sinphi;
  nv[2] = costheta;
  p[0] = cosphi * costheta;
  p[1] = sinphi * costheta;
  p[2] = -sintheta;
  // k = 1.;
  k = 1.;

  // p[0] = 1.;
  // p[1] = 0.;
  // p[2] = 0.;
  // phgPrintf("n[0]=%lf,n[1]=%lf,n[2]=%lf\n",n[0],n[1],n[2]);

  assert(p[0] * nv[0] + p[1] * nv[1] + p[2] * nv[2] == 0);
  GRID *g, *g_cell;
  DOF *E_h_re, *E_re, *E_inc_re;
  DOF *E_h_im, *E_im, *E_inc_im;
  DOF *curlE_inc_re, *curlE_inc_im;

  SOLVER *solver, *pc = NULL;

  // char *fn = "/share/home/yechangqing/machupeng/machupeng/grid/homo.mesh";
  // char *fn = "/share/home/clq/grid/cell.mesh";
  // char *fn = "/share/home/clq/grid/sphere.dat";
  // char *fn = "./sphere10.mesh";
  // char *fn = "./sphere-cube-fine.mesh";
  // char *fn_cell = "./cell.mesh";
  char *fn = "./sphere-cube22.mesh";

  char *vtk = NULL;
  INT pre_refines = 0;

  phgOptionsPreset(
      "-solver petsc  -oem_options { -ksp_type gmres -pc_type asm "
      "-pc_asm_overlap 4 -pc_asm_type restrict  -sub_ksp_type preonly "
      "-sub_pc_type lu  -sub_pc_factor_mat_solver_type  mumps -mat_mumps_sym 1 "
      "-mat_mumps_reicntl_4  0  -lag_summary -verbosity 1 }");

  phgOptionsRegisterInt("-pre_refines", "step num", &pre_refines);
  phgOptionsPreset("-dof_type ND1");
  //     phgOptionsPreset("-verbosity 1");
  // phgOptionsPreset("-solver_rtol 1.0e-8");

  phgInit(&argc, &argv);
  phgOptionsShowUsed();
  phgPrintf("n[0]=%lf,n[1]=%lf,n[2]=%lf,p[0]=%lf,p[1]=%lf,p[2]=%lf\n", nv[0],
            nv[1], nv[2], p[0], p[1], p[2]);
  g = phgNewGrid(-1);
  g_cell = phgNewGrid(-1);
  phgImportSetBdryMapFunc(bc_map);
  if (!phgImport(g, fn, FALSE))
    phgError(1, "can't read file \"%s\".\n", fn);

  phgCheckConformity(g);

  phgRefineAllElements(g, pre_refines);
  if (phgBalanceGrid(g, 1.2, -1, NULL, 0.))
    phgPrintf("Repartition mesh, lif: %lg\n\n", (double)g->lif);
  // phgExportMedit(g, "sphere.mesh");

  /* homo of q_re */

  /*===================================================*/

  E_inc_re = phgDofNew(g, DOF_ANALYTIC, 3, "E_inc_re", func_E_inc_re);
  E_inc_im = phgDofNew(g, DOF_ANALYTIC, 3, "E_inc_im", func_E_inc_im);

  curlE_inc_re = phgDofNew(g, DOF_ANALYTIC, 3, "E_inc_re", func_curlE_inc_re);
  curlE_inc_im = phgDofNew(g, DOF_ANALYTIC, 3, "E_inc_im", func_curlE_inc_im);

  DOF *f_refer_re, *f_refer_im, *f_homo_re, *f_homo_im;

  // phgDofMM(MAT_OP_N, MAT_OP_N, 3, 1, 3, 1.0,
  // theta_cube_re,-1,u_h_re, 1.0,&thetaXE_cube_re);
  // phgDofAXPBY(1,u_h_re,1.,&thetaXE_cube_re);

  // stime=Sqrt((diam_Min/3.));
  //	stime=0.05/2./2.;

  E_h_re = phgDofNew(g, DOF_DEFAULT, 1, "E_h_refer_re",
                     DofInterpolation /*DofNoAction*/);
  E_h_im = phgDofNew(g, DOF_DEFAULT, 1, "E_h_refer_im",
                     DofInterpolation /*DofNoAction*/);
  E_re = phgDofNew(g, DOF_P1, 3, "E_re", func_E_inc_re);
  E_im = phgDofNew(g, DOF_P1, 3, "E_im", func_E_inc_re);

  DOF *epsilon_re = phgDofNew(g, DOF_ANALYTIC, 1, NULL, func_epsilon_re);
  DOF *epsilon_im = phgDofNew(g, DOF_ANALYTIC, 1, NULL, func_epsilon_im);
  maxwell_forward_solver_etotal(E_h_re, E_h_im, E_inc_re, E_inc_im,
                                curlE_inc_re, curlE_inc_im, epsilon_re,
                                epsilon_im);
  phgPrintf("error re = %lf , error im = %lf \n", L2_Error(E_re, E_h_re),
            L2_Error(E_im, E_h_im));

  phgExportVTK(g, "g.vtk", E_h_re, E_h_im, E_re, E_im, NULL);
  // E_h_refer_re->DB_mask = 0;
  // E_h_refer_im->DB_mask = 0;

  // E_h_homo_re->DB_mask = 0;
  // E_h_homo_im->DB_mask = 0;
  // checkBdry(g);
  size_t mem;

  phgDofFree(&E_inc_re);

  phgDofFree(&E_inc_im);
  phgDofFree(&curlE_inc_re);

  phgDofFree(&curlE_inc_im);
  phgDofFree(&E_re);

  phgDofFree(&E_im);

  phgDofFree(&epsilon_re);

  phgDofFree(&epsilon_im);
  phgDofFree(&E_h_re);

  phgDofFree(&E_h_im);
  phgFreeGrid(&g);
  phgFinalize();

  return 0;
}
