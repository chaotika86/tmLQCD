/***********************************************************************
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008 Carsten Urbach
 *
 * This file is part of tmLQCD.
 *
 * tmLQCD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * tmLQCD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with tmLQCD.  If not, see <http://www.gnu.org/licenses/>.
 ***********************************************************************/
/**************************************************************************
 *
 * $Id$
 *
 * The externally accessible functions are
 *
 *   int bicgstab(spinor * const, spinor * const, const int, double, matrix_mult)
 *     BiCGstab solver
 * 
 *
 *
 *
 * Author: Carsten Urbach 
 *         <urbach@ifh.de>
 * 
 **************************************************************************/

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "global.h"
#include "su3.h"
#include "linalg_eo.h"
#include "start.h"
#include "bicgstab_complex.h"

/* P inout (guess for the solving spinor)
   Q input
*/
int bicgstab_complex(spinor * const P,spinor * const Q, const int max_iter, 
		     double eps_sq, const int rel_prec, 
		     const int N, matrix_mult f){
  double err, d1, squarenorm;
  complex rho0, rho1, omega, alpha, beta, nom, denom;
  int i;
  spinor * r, * p, * v, *hatr, * s, * t;

/*   init_solver_field(6); */
  hatr = g_spinor_field[DUM_SOLVER];
  r = g_spinor_field[DUM_SOLVER+1];
  v = g_spinor_field[DUM_SOLVER+2];
  p = g_spinor_field[DUM_SOLVER+3];
  s = g_spinor_field[DUM_SOLVER+4];
  t = g_spinor_field[DUM_SOLVER+5];

  f(r, P);
  diff(p, Q, r, N);
  assign(r, p, N);
  assign(hatr, p, N);
  rho0 = scalar_prod(hatr, r, N, 1);
  squarenorm = square_norm(Q, N, 1);

  for(i = 0; i < max_iter; i++){
    err = square_norm(r, N, 1);
    if(g_proc_id == g_stdio_proc && g_debug_level > 1) {
      printf("%d %e\n", i, err);
      fflush(stdout);
    }
  
    if((((err <= eps_sq) && (rel_prec == 0)) || ((err <= eps_sq*squarenorm) && (rel_prec == 1))) && i>0) {
      return(i);
    }
    f(v, p);
    denom = scalar_prod(hatr, v, N, 1);
    _div_complex(alpha, rho0, denom);
    assign(s, r, N);
    assign_diff_mul(s, v, alpha, N);
    f(t, s);
    omega = scalar_prod(t,s, N, 1);
    d1 = square_norm(t, N, 1);
    omega.re/=d1; omega.im/=d1;
    assign_add_mul_add_mul(P, p, s, alpha, omega, N);
    assign(r, s, N);
    assign_diff_mul(r, t, omega, N);
    rho1 = scalar_prod(hatr, r, N, 1);
    if(fabs(rho1.re) < 1.e-25 && fabs(rho1.im) < 1.e-25) {
      return(-1);
    }
    _mult_assign_complex(nom, alpha, rho1);
    _mult_assign_complex(denom, omega, rho0);
    _div_complex(beta, nom, denom);
    omega.re=-omega.re; omega.im=-omega.im;
    assign_mul_bra_add_mul_ket_add(p, v, r, omega, beta, N);
    rho0.re = rho1.re; rho0.im = rho1.im;
  }
  return -1;
}
