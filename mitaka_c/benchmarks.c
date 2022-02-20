#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "benchmarks.h"
#include "samplerZ.h"
#include "normaldist.h"
#include "api.h"
#include "poly.h"
#include "param.h"

#define BENCH_ITER 1000000
#define BENCH_ITER2 10000

static void random_poly(poly* p){
  for(int i=0; i < MITAKA_D; ++i)
    p->coeffs[i].v = (double)(rand()%MITAKA_Q);
}


void benchmark_FFT(){

  clock_t start = clock();
  
  poly p;
  random_poly(&p);

  for(int i=0; i < BENCH_ITER; ++i){
    FFT(&p);
  }
  clock_t stop = clock();
  double delta = (double)(stop - start) / CLOCKS_PER_SEC;
  printf("Elapsed time for %d FFT: %f. (%f ms per FTT)\n FFT/sec = %f\n", BENCH_ITER, delta, (delta/BENCH_ITER)*1000,BENCH_ITER/delta);

}

void benchmark_poly_add(){  

  clock_t start = clock();

  poly p1, p2;
  random_poly(&p1); random_poly(&p2);
  
  for(int i=0; i < BENCH_ITER; ++i){
    poly_add(&p1, &p2);
  }
  clock_t stop = clock();
  double delta = (double)(stop - start) / CLOCKS_PER_SEC;
  printf("Elapsed time for %d poly_add: %f. (%f ms per poly_add)\n poly_add/sec = %f\n", BENCH_ITER, delta, (delta/BENCH_ITER)*1000,BENCH_ITER/delta);

}

void benchmark_pointwise_mul(){

  clock_t start = clock();

  poly p1, p2;
  random_poly(&p1); random_poly(&p2);
  
  for(int i=0; i < BENCH_ITER; ++i){
    pointwise_mul(&p1, &p2);
  }
  clock_t stop = clock();
  double delta = (double)(stop - start) / CLOCKS_PER_SEC;
  printf("Elapsed time for %d pw_mul: %f. (%f ms per pw_mul)\n pw_mul/sec = %f\n", BENCH_ITER, delta, (delta/BENCH_ITER)*1000,BENCH_ITER/delta);


}

void benchmark_FFT_mul_adj(){

  clock_t start = clock();

  poly p1, p2;
  random_poly(&p1); random_poly(&p2);
  
  for(int i=0; i < BENCH_ITER; ++i){
    FFT_mul_adj(&p1,&p2);
  }
  clock_t stop = clock();
  double delta = (double)(stop - start) / CLOCKS_PER_SEC;
  printf("Elapsed time for %d mul_adj: %f. (%f ms per mul_adj)\n mul_adj/sec = %f\n", BENCH_ITER, delta, (delta/BENCH_ITER)*1000,BENCH_ITER/delta);


}

void benchmark_poly_div_FFT(){

  clock_t start = clock();

  poly p1, p2;
  random_poly(&p1); random_poly(&p2);
  
  for(int i=0; i < BENCH_ITER; ++i){
    poly_div_FFT(&p1, &p2);
  }
  clock_t stop = clock();
  double delta = (double)(stop - start) / CLOCKS_PER_SEC;
  printf("Elapsed time for %d poly_div: %f. (%f ms per poly_div)\n poly_div/sec = %f\n", BENCH_ITER, delta, (delta/BENCH_ITER)*1000,BENCH_ITER/delta);


}


void benchmark_sample_discrete_gauss(){

  clock_t start = clock();

  poly p;
  for(int i=0; i < MITAKA_D; ++i) p.coeffs[i].v = 0;
  for(int i=0; i < BENCH_ITER2; ++i){
    sample_discrete_gauss(&p);
  }
  clock_t stop = clock();
  double delta = (double)(stop - start) / CLOCKS_PER_SEC;
  printf("Elapsed time for %d discrete sampling: %f. (%f ms per sampling)\n sampling/sec = %f\n", BENCH_ITER2, delta, (delta/(BENCH_ITER2))*1000,(BENCH_ITER2)/delta);


}

void benchmark_normaldist(){

  clock_t start = clock();

  poly p;  
  for(int i=0; i < MITAKA_D; ++i) p.coeffs[i].v = 0;

  for(int i=0; i < BENCH_ITER2; ++i){
    normaldist(&p);
  }
  clock_t stop = clock();
  double delta = (double)(stop - start) / CLOCKS_PER_SEC;
  printf("Elapsed time for %d continuous sampling: %f. (%f ms per sampling)\n sampling/sec = %f\n", BENCH_ITER2, delta, (delta/(BENCH_ITER2))*1000,(BENCH_ITER2)/delta);


}


void run_benchmarks(){

  benchmark_FFT();
  benchmark_poly_add();
  benchmark_pointwise_mul();
  benchmark_FFT_mul_adj();
  benchmark_poly_div_FFT();
  benchmark_sample_discrete_gauss();
  benchmark_normaldist();
}
