#include "phg.h"
extern FLOAT k ;
extern FLOAT p[3];  // polarization
extern FLOAT nv[3];
void maxwell_forward_solver_etotal(DOF *E_h_re, DOF *E_h_im, DOF *E_inc_re,
                                   DOF *E_inc_im, DOF *curlE_inc_re,
                                   DOF *curlE_inc_im, DOF *epsilon_re,
                                   DOF *epsilon_im);
void maxwell_forward_solver(DOF *E_h_re, DOF *E_h_im, DOF *E_inc_re,
                            DOF *E_inc_im, DOF *q_re, DOF *q_im,
                            DOF *epsilon_re, DOF *epsilon_im);