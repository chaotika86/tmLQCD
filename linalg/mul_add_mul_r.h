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
/* $Id$*/

#ifndef _MUL_ADD_MUL_R_H
#define _MUL_ADD_MUL_R_H

#include "su3.h"

/* Makes (*R)=c1*(*S)+c2*(*U) , c1 and c2 are real constants */
void mul_add_mul_r(spinor * const R, spinor * const S, spinor * const U,
		   const double c1,const double c2, const int N);


#endif
