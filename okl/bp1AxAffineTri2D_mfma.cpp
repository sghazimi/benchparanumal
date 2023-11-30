#include <hip/hip_runtime.h>
/*

The MIT License (MIT)

Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

// Need dfloat == double
static_assert (sizeof(dfloat) == sizeof(double), "Dfloat must be double");

#if KERNEL_NUMBER==0

extern "C"
__global__
__launch_bounds__(64)
void bp1AxAffineTri2D(const dlong Nelements,
                      const  dlong  *  __restrict__ elementList,
                      const  dlong  *  __restrict__ GlobalToLocal,
                      const  dfloat *  __restrict__ wJ,
                      const  dfloat *  __restrict__ I,
                      const  dfloat *  __restrict__ MM,
                      const  dfloat *  __restrict__ q,
                             dfloat *  __restrict__ Aq){

  // The MFMA16x16x4 instruction computes a 16x16x4 matrix
  //  multiplication using a single wavefront.

  //The result of the instruction is a 16x16 matrix, stored in a double4 in each lane
  using double4 = __attribute__((__vector_size__(4 * sizeof(double)))) double;

  //This kernel uses only one wavefront, and the number of DOFs must be <=16
  static_assert (p_Np <= 16, "Kernel only supports small element right now");

  dfloat  s_MM[p_Np][p_Np];
  dfloat  s_Aq[p_Np*4];

  //To pack the M*q operation into a matrix-matrix multiply, we must pack multiple q
  // elements into this wavefront. Since the instruction inputs are 16x4, we will process
  // 16 elements on this wave
  const dlong eo = 16*blockIdx.x;

  //We launch a (16,4) size threadblock
  const dlong ei = threadIdx.x;
  dlong n = threadIdx.y; 

  const dlong e = eo+ei;
  const dlong element = (e<Nelements) ? elementList[e] : -1;

  double4 Mq = {0}; // zero out 16x16 result
  
  int iters=(p_Np/4)+1;
  int count=0;

  while(count<iters) {

  int nt=n+count*4;

  s_MM[nt][ei] = MM[ei + nt*p_Np];

  const dlong id = (nt<p_Np && element!=-1) ? GlobalToLocal[nt + element*p_Np] : -1;

  const dfloat r_q = (id!=-1) ? q[id] : 0;

  //const dfloat r_MM = (nt<p_Np && ei<p_Np) ? MM[ei + nt*p_Np] : 0;
  const dfloat r_MM = (nt<p_Np && ei<p_Np) ? s_MM[nt][ei] : 0;

  Mq = __builtin_amdgcn_mfma_f64_16x16x4f64(r_MM, r_q, Mq, 0, 0, 0);

  count+=1;
  }

  if(element!=-1) { 
  const dfloat J = wJ[element];

  count=0;
  while(count<iters) {
  int nt=n+count*4;
  const dlong base = (nt< p_Np) ? nt + element*p_Np : -1;
  if(base != -1) {
     s_Aq[base] = J*Mq[count];
     Aq[base]=s_Aq[base];
  }
  count+=1;
  }
  }
}
#endif

#if KERNEL_NUMBER==1

extern "C"
__global__
__launch_bounds__(64)
void bp1AxAffineTri2D(const dlong Nelements,
                      const  dlong  *  __restrict__ elementList,
                      const  dlong  *  __restrict__ GlobalToLocal,
                      const  dfloat *  __restrict__ wJ,
                      const  dfloat *  __restrict__ I,
                      const  dfloat *  __restrict__ MM,
                      const  dfloat *  __restrict__ q,
                             dfloat *  __restrict__ Aq){

  // The MFMA4x4x4 instruction computes 4x 4x4x4 matrix
  //  multiplications using a single wavefront.

  //The result of the instruction is 4x 4x4 matrices, stored in a double in each lane

  //This kernel uses only one wavefront, and the number of DOFs must be <=16
  static_assert (p_Np <= 4, "Kernel only supports small element right now");

  //To pack the M*q operation into a matrix-matrix multiply, we must pack multiple q
  // elements into this wavefront. Since the instruction inputs are 4x 4x4, we will process
  // 16 elements on this wave

  const dlong eo = 16*blockIdx.x;

  //We launch a (16,4) size threadblock
  const dlong ei = threadIdx.x; //different lanes read different elements
  dlong n = threadIdx.y; //second dimension is DOFs

  const dlong e = eo+ei;
  const dlong element = (e<Nelements) ? elementList[e] : -1;

  dfloat Mq = {0}; // zero out result
 
  const dlong id = (n<p_Np && element!=-1) ? GlobalToLocal[n + element*p_Np] : -1;
  const dfloat r_q = (id!=-1) ? q[id] : 0;
  const dfloat r_MM = (n<p_Np && (ei%4)<p_Np) ? MM[(ei%4) + n*p_Np] : 0;

  Mq = __builtin_amdgcn_mfma_f64_4x4x4f64(r_MM, r_q, Mq, 0, 0, 0);

  if(element!=-1) {
  const dfloat J = wJ[element];

  const dlong base = (n< p_Np) ? n + element*p_Np : -1;
  if(base != -1) {
     Aq[base] = J*Mq;
  }
  }
}
#endif
