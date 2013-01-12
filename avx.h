/*
 *
 *	AVX.h: support routines for AVX1-based hopping matrix implementation
 *
 *
 */

#ifndef __AVX_H
#define __AVX_H

/*
	Functions ported from SSE/SSE2/SSE3 needed for HoppingMatrix (v5)
	-------------------------------------------------------------------------------
		_sse_load							Done
		_sse_load_up						Done
		_sse_store							Done
		_sse_store_nt						
		_sse_store_up						Done
		_sse_vector_add						Done
		_sse_vector_sub						Done
		_sse_su3_multiply					Done
		_sse_su3_inverse_multiply			Done
		_sse_vector_cmplx_mul				Done
		_sse_vector_cmplxcg_mul				Done
		_sse_vector_i_mul					Done
		
		_sse_vector_sub_up
		
*/

typedef struct avx_int {
	int32_t c0, c1, c2, c3,
			c4, c5, c6, c7;
} avx_int __attribute__((aligned (32)));
static avx_int _avx_sgn __attribute__ ((unused)) ={0,0x80000000,0,0,0,0,0,0};


// Prefetch logic (directly borrowed from SSE3)
#ifdef DISABLE_PREFETCH
	#define _prefetch_spinor(addr)
	#define _prefetch_nta_spinor(addr)
	#define _prefetch_halfspinor(addr)
	#define _prefetch_nta_halfspinor(addr)
	#define _prefetch_su3(addr)
	#define _prefetch_mom(addr)
#else
	#define _prefetch_spinor(addr) \
	__asm__ __volatile__ ("prefetcht0 %0 \n\t" \
		                  "prefetcht0 %1 \n\t" \
		                  "prefetcht0 %2" \
		                  : \
		                  : \
		                  "m" (*(((char*)(addr)))), \
		                  "m" (*(((char*)(addr))+64)),			\
		                  "m" (*(((char*)(addr))+128)))


	#define _prefetch_nta_spinor(addr) \
	__asm__ __volatile__ ("prefetchnta %0 \n\t" \
		                  "prefetchnta %1 \n\t" \
		                  "prefetchnta %2" \
		                  : \
		                  : \
		                  "m" (*(((char*)(addr)))), \
		                  "m" (*(((char*)(addr))+64)), \
		                  "m" (*(((char*)(addr))+128)))

	#define _prefetch_halfspinor(addr) \
	__asm__ __volatile__ ("prefetcht0 %0 \n\t" \
				  "prefetcht0 %1" \
		                  : \
		                  : \
		                  "m" (*(((char*)(addr)))), \
		                  "m" (*(((char*)(addr))+64)))


	#define _prefetch_nta_halfspinor(addr) \
	__asm__ __volatile__ ("prefetchnta %0 \n\t" \
				  "prefetchnta %1" \
		                  : \
		                  : \
		                  "m" (*(((char*)(addr)))), \
		                  "m" (*(((char*)(addr))+64)))

	#define _prefetch_su3(addr) \
	__asm__ __volatile__ ("prefetcht0 %0  \n\t" \
		                  "prefetcht0 %1  \n\t" \
		                  "prefetcht0 %2" \
		                  : \
		                  : \
		                  "m" (*(((char*)(addr)))), \
		                  "m" (*(((char*)(addr))+64)), \
		                  "m" (*(((char*)(addr))+128)))

	#define _prefetch_mom(addr) \
	__asm__ __volatile__ ("prefetcht0 %0" \
		                  : \
		                  : \
		                  "m" (*(((char*)((addr))))))
#endif




/*
* Loads an su3 vector s to ymm0,ymm1,ymm2
*/

#define _avx_load(s) \
	__asm__ __volatile__ ("vbroadcastf128 %0, %%ymm0 \n\t" \
		                  "vbroadcastf128 %1, %%ymm1 \n\t" \
		                  "vbroadcastf128 %2, %%ymm2" \
		                  : \
		                  : \
		                  "m" ((s).c0), \
		                  "m" ((s).c1), \
		                  "m" ((s).c2))

/*
* Loads an su3 vector s to ymm3,ymm4,ymm5
*/  

#define _avx_load_up(s) \
	__asm__ __volatile__ ("vbroadcastf128 %0, %%ymm3 \n\t" \
		                  "vbroadcastf128 %1, %%ymm4 \n\t" \
		                  "vbroadcastf128 %2, %%ymm5" \
		                  : \
		                  : \
		                  "m" ((s).c0), \
		                  "m" ((s).c1), \
		                  "m" ((s).c2))

/*
* Stores xmm0,xmm1,xmm2 to the components r.c1,r.c2,r.c3 of an su3 vector
*/

#define _avx_store(r) \
	__asm__ __volatile__ ("vmovupd %%xmm0, %0 \n\t" \
		                  "vmovupd %%xmm1, %1 \n\t" \
		                  "vmovupd %%xmm2, %2" \
		                  : \
		                  "=m" ((r).c0), \
		                  "=m" ((r).c1), \
		                  "=m" ((r).c2))

/*
* Stores xmm3,xmm4,xmm5 to the components r.c1,r.c2,r.c3 of an su3 vector
*/

#define _avx_store_up(r) \
	__asm__ __volatile__ ("vmovupd %%xmm3, %0 \n\t" \
		                  "vmovupd %%xmm4, %1 \n\t" \
		                  "vmovupd %%xmm5, %2" \
		                  : \
		                  "=m" ((r).c0), \
		                  "=m" ((r).c1), \
		                  "=m" ((r).c2))

/*
* Stores xmm0,xmm1,xmm2 to the components r.c1,r.c2,r.c3 of an su3 vector
*/

#define _avx_store_nt(r) \
	__asm__ __volatile__ ("vmovntpd %%xmm0, %0 \n\t" \
		                  "vmovntpd %%xmm1, %1 \n\t" \
		                  "vmovntpd %%xmm2, %2" \
		                  : \
		                  "=m" ((r).c0), \
		                  "=m" ((r).c1), \
		                  "=m" ((r).c2))
                      
#define _avx_store_nt_up(r) \
__asm__ __volatile__ ("vmovntpd %%ymm3, %0 \n\t" \
                      "vmovntpd %%ymm4, %1 \n\t" \
                      "vmovntpd %%ymm5, %2" \
                      : \
                      "=m" ((r).c0), \
                      "=m" ((r).c1), \
                      "=m" ((r).c2))

/*
* Adds ymm3,ymm4,ymm5 to ymm0,ymm1,ymm2
*/

#define _avx_vector_add() \
	__asm__ __volatile__ ("vaddpd %%ymm3, %%ymm0, %%ymm0 \n\t" \
		                  "vaddpd %%ymm4, %%ymm1, %%ymm1 \n\t" \
		                  "vaddpd %%ymm5, %%ymm2, %%ymm2" \
		                  : \
		                  :)


/*
* Subtracts ymm3,ymm4,ymm5 from ymm0,ymm1,ymm2
*/

#define _avx_vector_sub() \
	__asm__ __volatile__ ("vsubpd %%ymm3, %%ymm0, %%ymm0 \n\t" \
		                  "vsubpd %%ymm4, %%ymm1, %%ymm1 \n\t" \
		                  "vsubpd %%ymm5, %%ymm2, %%ymm2" \
		                  : \
		                  :)
		                  
#define _avx_vector_sub_up() \
	__asm__ __volatile__ ("vsubpd %%ymm0, %%ymm3, %%ymm3 \n\t" \
		                  "vsubpd %%ymm1, %%ymm4, %%ymm4 \n\t" \
		                  "vsubpd %%ymm2, %%ymm5, %%ymm5" \
		                  : \
		                  :)



/*
* Multiplies an su3 vector s with an su3 matrix u, assuming s is
* stored in  ymm0,ymm1,ymm2
*
* On output the result is in xmm3,xmm4,xmm5 and the registers 
* ymm0,ymm1,ymm2 are changed
*/

#define _avx_su3_multiply(u) \
    __asm__ __volatile__ (  /* initialize registers */ \
					"vmovddup       %%ymm0, %%ymm3  \n\t" \
					"vmovddup       %%ymm1, %%ymm4  \n\t" \
					"vmovddup       %%ymm2, %%ymm5  \n\t" \
					/* calculate real part */ \
					/* first column */ \
					"vinsertf128    $0x0,   %0,     %%ymm6, %%ymm6  \n\t" \
					"vinsertf128    $0x1,   %1,     %%ymm6, %%ymm6  \n\t" \
					"vinsertf128    $0x0,   %2,     %%ymm7, %%ymm7  \n\t" \
					"vmulpd         %%ymm6, %%ymm3, %%ymm12  \n\t" \
					"vmulpd         %%ymm7, %%ymm3, %%ymm13  \n\t" \
					/* second column */ \
					"vinsertf128    $0x0,   %3,     %%ymm8, %%ymm8  \n\t" \
					"vinsertf128    $0x1,   %4,     %%ymm8, %%ymm8  \n\t" \
					"vinsertf128    $0x0,   %5,     %%ymm9, %%ymm9  \n\t" \
					"vmulpd         %%ymm8, %%ymm4, %%ymm14 \n\t" \
					"vaddpd         %%ymm12,%%ymm14,%%ymm12 \n\t" \
					"vmulpd         %%ymm9, %%ymm4, %%ymm14 \n\t" \
					"vaddpd         %%ymm13,%%ymm14,%%ymm13 \n\t" \
					/* third column */ \
					"vinsertf128    $0x0,   %6,     %%ymm10, %%ymm10  \n\t" \
					"vinsertf128    $0x1,   %7,     %%ymm10, %%ymm10  \n\t" \
					"vinsertf128    $0x0,   %8,     %%ymm11, %%ymm11  \n\t" \
					"vmulpd         %%ymm10,%%ymm5, %%ymm14 \n\t" \
					"vaddpd         %%ymm12,%%ymm14,%%ymm12 \n\t" \
					"vmulpd         %%ymm11,%%ymm5, %%ymm14 \n\t" \
					"vaddpd         %%ymm13,%%ymm14,%%ymm13 \n\t" \
					/* calculate imaginary part */ \
					/* swap parts */ \
					"vpermilpd      $0x5,   %%ymm6, %%ymm6 \n\t" \
					"vpermilpd      $0x5,   %%ymm7, %%ymm7 \n\t" \
					"vpermilpd      $0x5,   %%ymm8, %%ymm8 \n\t" \
					"vpermilpd      $0x5,   %%ymm9, %%ymm9 \n\t" \
					"vpermilpd      $0x5,   %%ymm10, %%ymm10 \n\t" \
					"vpermilpd      $0x5,   %%ymm11, %%ymm11 \n\t" \
					/* initialize registers */ \
					"vpermilpd     $0x5,    %%ymm0, %%ymm0  \n\t" \
					"vpermilpd     $0x5,    %%ymm1, %%ymm1  \n\t" \
					"vpermilpd     $0x5,    %%ymm2, %%ymm2  \n\t" \
					"vmovddup      %%ymm0,  %%ymm3  \n\t" \
					"vmovddup      %%ymm1,  %%ymm4  \n\t" \
					"vmovddup      %%ymm2,  %%ymm5  \n\t" \
					/* first column */ \
					"vmulpd         %%ymm6, %%ymm3, %%ymm0  \n\t" \
					"vmulpd         %%ymm7, %%ymm3, %%ymm1  \n\t" \
					/* second column */ \
					"vmulpd         %%ymm8, %%ymm4, %%ymm2 \n\t" \
					"vaddpd         %%ymm0, %%ymm2, %%ymm0 \n\t" \
					"vmulpd         %%ymm9, %%ymm4, %%ymm2 \n\t" \
					"vaddpd         %%ymm1, %%ymm2, %%ymm1 \n\t" \
					/* third column */ \
					"vmulpd         %%ymm10,%%ymm5, %%ymm2 \n\t" \
					"vaddpd         %%ymm0, %%ymm2, %%ymm0 \n\t" \
					"vmulpd         %%ymm11,%%ymm5, %%ymm2 \n\t" \
					"vaddpd         %%ymm1, %%ymm2, %%ymm1 \n\t" \
					/* final touches */ \
					"vaddsubpd      %%ymm0,%%ymm12, %%ymm0 \n\t" \
					"vaddsubpd      %%ymm1,%%ymm13, %%ymm1 \n\t" \
					/* extract final vectors */ \
					"vextractf128   $0x0,   %%ymm0, %%xmm3 \n\t" \
					"vextractf128   $0x1,   %%ymm0, %%xmm4 \n\t" \
					"vextractf128   $0x0,   %%ymm1, %%xmm5 \n\t" \
					: \
					: \
					"m"((u).c00), "m"((u).c10), "m"((u).c20), \
					"m"((u).c01), "m"((u).c11), "m"((u).c21), \
					"m"((u).c02), "m"((u).c12), "m"((u).c22))

/*
* Multiplies xmm3,xmm4,xmm5 with the complex number c
*/
#define _avx_vector_cmplx_mul(c) \
	__asm__ __volatile__ ( /* compactify registers */ \
					"vinsertf128	$0x0,	%%xmm3,	%%ymm6,	%%ymm6	\n\t" \
					"vinsertf128	$0x1,	%%xmm4,	%%ymm6,	%%ymm6	\n\t" \
					"vmovapd		%%ymm5,	%%ymm7	\n\t" \
					/* load data in cache */ \
					"vbroadcastf128	%0, 	%%ymm8	\n\t" \
					"vmovddup		%%ymm8,	%%ymm9	\n\t" \
					/* get real part */ \
					"vmulpd			%%ymm9,	%%ymm6,	%%ymm10	\n\t" \
					"vmulpd			%%ymm9,	%%ymm7,	%%ymm11	\n\t" \
					/* prepare registers */ \
					"vpermilpd		$0x5,	%%ymm6,	%%ymm6	\n\t" \
					"vpermilpd		$0x5,	%%ymm7,	%%ymm7	\n\t" \
					"vpermilpd		$0x5,	%%ymm8,	%%ymm8	\n\t" \
					"vmovddup		%%ymm8,	%%ymm9	\n\t" \
					/* get imaginary part */ \
					"vmulpd			%%ymm9,	%%ymm6,	%%ymm12	\n\t" \
					"vmulpd			%%ymm9,	%%ymm7,	%%ymm13	\n\t" \
					/* final touches */ \
					"vaddsubpd		%%ymm12, %%ymm10, %%ymm10 \n\t" \
					"vaddsubpd		%%ymm13, %%ymm11, %%ymm11 \n\t" \
					/* extract stuff */ \
					"vextractf128 	$0x0,  	%%ymm10, %%xmm3	\n\t" \
					"vextractf128 	$0x1,	%%ymm10, %%xmm4	\n\t" \
					"vextractf128 	$0x0,	%%ymm11, %%xmm5 \n\t" \
					: \
					: \
					"m"(c) )

/*
 * Multiplies xmm3,xmm4,xmm5 with the complex 
 * conjugate of the number c
 */
#define _avx_vector_cmplxcg_mul(c) \
	__asm__ __volatile__ ("vbroadcastf128 	%0, %%ymm0	\n\t" \
						  "vpermilpd $0x5, %%ymm0, %%ymm0 \n\t" \
						  /* pack vectors */ \
						  "vinsertf128 $0x1, %%xmm4, %%ymm3, %%ymm3 \n\t" \
						  /* copy imag part */ \
						  "vmovddup %%ymm0, %%ymm1 \n\t" \
						  /* multiply it */ \
						  "vmulpd %%ymm1, %%ymm3, %%ymm6 \n\t" \
						  "vmulpd %%ymm1, %%ymm5, %%ymm7 \n\t" \
						  /* copy real part */ \
						  "vpermilpd $0x5, %%ymm0, %%ymm0 \n\t" \
						  "vmovddup %%ymm0, %%ymm1 \n\t" \
						  /* permute base vectors */ \
						  "vpermilpd $0x5, %%ymm3, %%ymm3 \n\t" \
						  "vpermilpd $0x5, %%ymm5, %%ymm5 \n\t" \
						  /* multiply 'em */ \
						  "vmulpd %%ymm1, %%ymm3, %%ymm3 \n\t" \
						  "vmulpd %%ymm1, %%ymm5, %%ymm5 \n\t" \
						  /* addsub */ \
						  "vaddsubpd %%ymm6, %%ymm3, %%ymm3 \n\t" \
						  "vaddsubpd %%ymm7, %%ymm5, %%ymm5 \n\t" \
						  /* xchange and extract */ \
						  "vpermilpd $0x5, %%ymm3, %%ymm3 \n\t" \
						  "vpermilpd $0x5, %%ymm5, %%ymm5 \n\t" \
						  "vextractf128 $0x1, %%ymm3, %%xmm4 \n\t" \
						  : \
						  : \
						  "m" ((c)) )

/*
* Multiplies ymm3,ymm4,ymm5 with i
*/

#define _avx_vector_i_mul() \
	__asm__ __volatile__ ("vshufpd $0x1, %%ymm3, %%ymm3, %%ymm3 \n\t" \
				          "vshufpd $0x1, %%ymm4, %%ymm4, %%ymm4 \n\t" \
				          "vshufpd $0x1, %%ymm5, %%ymm5, %%ymm5 \n\t" \
				          "vxorpd %0, %%ymm3, %%ymm3 \n\t" \
				          "vxorpd %0, %%ymm4, %%ymm4 \n\t" \
				          "vxorpd %0, %%ymm5, %%ymm5" \
				          : \
				          : \
				          "m" (_avx_sgn))


/*
 * Multiplies an su3 vector s with an su3 matrix u^dagger, assuming s is
 * stored in  xmm0,xmm1,xmm2
 *
 * On output the result is in xmm3,xmm4,xmm5 and the registers 
 * xmm0,xmm1,xmm2 are changed
 */
#define _avx_su3_inverse_multiply(u) \
    __asm__ __volatile__ (  /* initialize registers */ \
					"vpermilpd     $0x5,    %%ymm0, %%ymm0  \n\t" \
					"vpermilpd     $0x5,    %%ymm1, %%ymm1  \n\t" \
					"vpermilpd     $0x5,    %%ymm2, %%ymm2  \n\t" \
					"vmovddup       %%ymm0, %%ymm3  \n\t" \
					"vmovddup       %%ymm1, %%ymm4  \n\t" \
					"vmovddup       %%ymm2, %%ymm5  \n\t" \
					/* calculate imaginary part */ \
					/* first column */ \
					"vinsertf128    $0x0,   %0,     %%ymm6, %%ymm6  \n\t" \
					"vinsertf128    $0x1,   %1,     %%ymm6, %%ymm6  \n\t" \
					"vinsertf128    $0x0,   %2,     %%ymm7, %%ymm7  \n\t" \
					"vmulpd         %%ymm6, %%ymm3, %%ymm12  \n\t" \
					"vmulpd         %%ymm7, %%ymm3, %%ymm13  \n\t" \
					/* second column */ \
					"vinsertf128    $0x0,   %3,     %%ymm8, %%ymm8  \n\t" \
					"vinsertf128    $0x1,   %4,     %%ymm8, %%ymm8  \n\t" \
					"vinsertf128    $0x0,   %5,     %%ymm9, %%ymm9  \n\t" \
					"vmulpd         %%ymm8, %%ymm4, %%ymm14 \n\t" \
					"vaddpd         %%ymm12,%%ymm14,%%ymm12 \n\t" \
					"vmulpd         %%ymm9, %%ymm4, %%ymm14 \n\t" \
					"vaddpd         %%ymm13,%%ymm14,%%ymm13 \n\t" \
					/* third column */ \
					"vinsertf128    $0x0,   %6,     %%ymm10, %%ymm10  \n\t" \
					"vinsertf128    $0x1,   %7,     %%ymm10, %%ymm10  \n\t" \
					"vinsertf128    $0x0,   %8,     %%ymm11, %%ymm11  \n\t" \
					"vmulpd         %%ymm10,%%ymm5, %%ymm14 \n\t" \
					"vaddpd         %%ymm12,%%ymm14,%%ymm12 \n\t" \
					"vmulpd         %%ymm11,%%ymm5, %%ymm14 \n\t" \
					"vaddpd         %%ymm13,%%ymm14,%%ymm13 \n\t" \
					/* calculate real part */ \
					/* swap parts */ \
					"vpermilpd      $0x5,   %%ymm6, %%ymm6 \n\t" \
					"vpermilpd      $0x5,   %%ymm7, %%ymm7 \n\t" \
					"vpermilpd      $0x5,   %%ymm8, %%ymm8 \n\t" \
					"vpermilpd      $0x5,   %%ymm9, %%ymm9 \n\t" \
					"vpermilpd      $0x5,   %%ymm10, %%ymm10 \n\t" \
					"vpermilpd      $0x5,   %%ymm11, %%ymm11 \n\t" \
					/* initialize registers */ \
					"vpermilpd     $0x5,    %%ymm0, %%ymm0  \n\t" \
					"vpermilpd     $0x5,    %%ymm1, %%ymm1  \n\t" \
					"vpermilpd     $0x5,    %%ymm2, %%ymm2  \n\t" \
					"vmovddup      %%ymm0,  %%ymm3  \n\t" \
					"vmovddup      %%ymm1,  %%ymm4  \n\t" \
					"vmovddup      %%ymm2,  %%ymm5  \n\t" \
					/* first column */ \
					"vmulpd         %%ymm6, %%ymm3, %%ymm0  \n\t" \
					"vmulpd         %%ymm7, %%ymm3, %%ymm1  \n\t" \
					/* second column */ \
					"vmulpd         %%ymm8, %%ymm4, %%ymm2 \n\t" \
					"vaddpd         %%ymm0, %%ymm2, %%ymm0 \n\t" \
					"vmulpd         %%ymm9, %%ymm4, %%ymm2 \n\t" \
					"vaddpd         %%ymm1, %%ymm2, %%ymm1 \n\t" \
					/* third column */ \
					"vmulpd         %%ymm10,%%ymm5, %%ymm2 \n\t" \
					"vaddpd         %%ymm0, %%ymm2, %%ymm0 \n\t" \
					"vmulpd         %%ymm11,%%ymm5, %%ymm2 \n\t" \
					"vaddpd         %%ymm1, %%ymm2, %%ymm1 \n\t" \
					/* final touches */ \
					"vaddsubpd      %%ymm0,%%ymm12, %%ymm0 \n\t" \
					"vaddsubpd      %%ymm1,%%ymm13, %%ymm1 \n\t" \
					/* invert */ \
					"vpermilpd		$0x5, 	%%ymm0,	%%ymm0 \n\t" \
					"vpermilpd		$0x5,	%%ymm1, %%ymm1 \n\t" \
					/* extract final vectors */ \
					"vextractf128   $0x0,   %%ymm0, %%xmm3 \n\t" \
					"vextractf128   $0x1,   %%ymm0, %%xmm4 \n\t" \
					"vextractf128   $0x0,   %%ymm1, %%xmm5 \n\t" \
					: \
					: \
					"m"((u).c00), "m"((u).c01), "m"((u).c02), \
					"m"((u).c10), "m"((u).c11), "m"((u).c12), \
					"m"((u).c20), "m"((u).c21), "m"((u).c22))

#endif
