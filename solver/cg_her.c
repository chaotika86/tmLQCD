/***********************************************************************
 * Copyright (C) 2001 Martin Hasenbusch
 *               2003 Thomas Chiarappa
 *               2002,2003,2004,2005 Carsten Urbach
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
 * File: cg_her.c
 *
 * CG solver for hermitian f only!
 *
 * The externally accessible functions are
 *
 *
 *   int cg(spinor * const P, spinor * const Q, double m, const int subtract_ev)
 *     CG solver
 *
 * input:
 *   m: Mass to be use in D_psi
 *   subtrac_ev: if set to 1, the lowest eigenvectors of Q^2 will
 *               be projected out.
 *   Q: source
 * inout:
 *   P: initial guess and result
 * 
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
#include "solver/matrix_mult_typedef.h"
#include "sub_low_ev.h"
#include "poly_precon.h"
#include "cg_her.h"

/* P output = solution , Q input = source */
int cg_her(spinor * const P, spinor * const Q, const int max_iter, 
	   double eps_sq, const int rel_prec, const int N, matrix_mult f, 
	   const int subtract_ev, const int modulo){
  double normsp, normsq, pro, err, alpha_cg, beta_cg, squarenorm;
  int iteration;
  
  squarenorm = square_norm(Q, N, 1);
  /*        !!!!   INITIALIZATION    !!!! */
  assign(g_spinor_field[DUM_SOLVER], P, N);
  /*        (r_0,r_0)  =  normsq         */
  normsp=square_norm(P, N, 1);

  if((subtract_ev == 1)) { 
    assign_sub_lowest_eigenvalues(g_spinor_field[DUM_SOLVER+5], Q, 10, N);
  }
  else{
    assign(g_spinor_field[DUM_SOLVER+5], Q, N);
  }
  
  /* initialize residue r and search vector p */
  if(normsp==0){
    /* if a starting solution vector equal to zero is chosen */
    assign(g_spinor_field[DUM_SOLVER+1], g_spinor_field[DUM_SOLVER+5], N);
    assign(g_spinor_field[DUM_SOLVER+2], g_spinor_field[DUM_SOLVER+5], N);
    normsq=square_norm(Q, N, 1);
  }
  else{
    /* if a starting solution vector different from zero is chosen */
    f(g_spinor_field[DUM_SOLVER+3], g_spinor_field[DUM_SOLVER]);
   
    if((subtract_ev == 1)) {
      sub_lowest_eigenvalues(g_spinor_field[DUM_SOLVER+3], g_spinor_field[DUM_SOLVER], 10, N);
    }
    diff(g_spinor_field[DUM_SOLVER+1], g_spinor_field[DUM_SOLVER+5], g_spinor_field[DUM_SOLVER+3], N);
    assign(g_spinor_field[DUM_SOLVER+2], g_spinor_field[DUM_SOLVER+1], N);
    normsq=square_norm(g_spinor_field[DUM_SOLVER+2], N, 1);
  }
  
  /* main loop */
  for(iteration=0;iteration<max_iter;iteration++){
    f(g_spinor_field[DUM_SOLVER+4], g_spinor_field[DUM_SOLVER+2]);

    if((subtract_ev == 1) && (iteration%modulo == 0)) {
      sub_lowest_eigenvalues(g_spinor_field[DUM_SOLVER+4], g_spinor_field[DUM_SOLVER+2], 10, N);
    }
    /* c=scalar_prod(&g_ev[0*VOLUME], g_spinor_field[DUM_SOLVER+4], 1);
       printf("%e, %e\n",c.re,c.im); */
    pro=scalar_prod_r(g_spinor_field[DUM_SOLVER+2], g_spinor_field[DUM_SOLVER+4], N, 1);
     
    /*  Compute alpha_cg(i+1)   */
    alpha_cg=normsq/pro;
     
    /*  Compute x_(i+1) = x_i + alpha_cg(i+1) p_i    */
    assign_add_mul_r(g_spinor_field[DUM_SOLVER], g_spinor_field[DUM_SOLVER+2],  alpha_cg, N);
    /*  Compute r_(i+1) = r_i - alpha_cg(i+1) Qp_i   */
    assign_add_mul_r(g_spinor_field[DUM_SOLVER+1], g_spinor_field[DUM_SOLVER+4], -alpha_cg, N);

    /* Check whether the precision is reached ... */
    err=square_norm(g_spinor_field[DUM_SOLVER+1], N, 1);
    if(g_debug_level > 0 && g_proc_id == g_stdio_proc) {
      printf("%d\t%g\n",iteration,err); fflush( stdout);
    }

    if(((err <= eps_sq) && (rel_prec == 0)) || ((err <= eps_sq*squarenorm) && (rel_prec == 1))) {
      if((subtract_ev == 1)){
	assign_add_invert_subtracted_part(g_spinor_field[DUM_SOLVER], Q, 10, N);
      } 
      assign(P, g_spinor_field[DUM_SOLVER], N);
      f(g_spinor_field[DUM_SOLVER+2], P);
      diff(g_spinor_field[DUM_SOLVER+3], g_spinor_field[DUM_SOLVER+2], Q, N);
      err = square_norm(g_spinor_field[DUM_SOLVER+3], N, 1);
      if(g_debug_level > 0 && g_proc_id == g_stdio_proc) {
	printf("true residue %d\t%g\t\n",iteration, err); fflush( stdout);
      }
      g_sloppy_precision = 0;
      return(iteration+1);
    }
#ifdef _USE_HALFSPINOR
    if(((err*err <= eps_sq) && (rel_prec == 0)) || ((err*err <= eps_sq*squarenorm) && (rel_prec == 1))) {
      g_sloppy_precision = 1;
      if(g_debug_level > 2 && g_proc_id == g_stdio_proc) {
	printf("sloppy precision on\n"); fflush( stdout);
      }
    }
#endif
    /* Compute beta_cg(i+1)
       Compute p_(i+1) = r_i+1 + beta_(i+1) p_i     */
    beta_cg=err/normsq;
    assign_mul_add_r(g_spinor_field[DUM_SOLVER+2], beta_cg, g_spinor_field[DUM_SOLVER+1], N);
    normsq=err;
  }
  if((subtract_ev == 1)) { 
    assign_add_invert_subtracted_part(g_spinor_field[DUM_SOLVER], Q, 10, N);
  }
  assign(P, g_spinor_field[DUM_SOLVER], N);
  g_sloppy_precision = 0;
  return(-1);
}

static char const rcsid[] = "$Id$";








