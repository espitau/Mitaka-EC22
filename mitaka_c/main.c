#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "api.h"
#include "poly.h"
#include "test_dist.h"
#include "benchmarks.h"


#include "precomp.h"
#include "normaldist.h"
#include "samplerZ.h"
#include "randombytes.h"

#define PRINT_START_END(str,x) do { \
    	printf("* %s: ", str); \
	for(int i=0; i < PRINT_SIZE; ++i) \
    	  printf("%f%+fj  ", (x).coeffs[i].v, (x).coeffs[i+MITAKA_D/2].v); \
    	printf("... "); \
	for(int i=MITAKA_D/2 - PRINT_SIZE; i < MITAKA_D/2; ++i) \
    	  printf("%f%+fj  ", (x).coeffs[i].v, (x).coeffs[i+MITAKA_D/2].v); \
    	printf("\n"); \
    	} while(0)

static void print_sk(secret_key* sk){
  int PRINT_SIZE = 8;//MITAKA_D;

  PRINT_START_END("b10", sk->b10);
  PRINT_START_END("b11", sk->b11);
  PRINT_START_END("b20", sk->b20);
  PRINT_START_END("b21", sk->b21);
  PRINT_START_END("b21", sk->b21);
  PRINT_START_END("GSO_b10", sk->GSO_b10);
  PRINT_START_END("GSO_b11", sk->GSO_b11);
  PRINT_START_END("GSO_b20", sk->GSO_b20);
  PRINT_START_END("GSO_b21", sk->GSO_b21);
  PRINT_START_END("sigma1", sk->sigma1);
  PRINT_START_END("sigma2", sk->sigma2);
  PRINT_START_END("beta10", sk->beta10);
  PRINT_START_END("beta11", sk->beta11);
  PRINT_START_END("beta20", sk->beta20);
  PRINT_START_END("beta21", sk->beta21);

}





void speed(){
  secret_key sk;
  public_key pk;
  signature s;
  precomp(&sk);
  load_pk(&pk);


  uint8_t m[32] = {0x46,0xb6,0xc4,0x83,0x3f,0x61,0xfa,0x3e,0xaa,0xe9,0xad,0x4a,0x68,0x8c,0xd9,0x6e,0x22,0x6d,0x93,0x3e,0xde,0xc4,0x64,0x9a,0xb2,0x18,0x45,0x2,0xad,0xf3,0xc,0x61};


  int iter = 10000;
  clock_t start = clock();

  for(int i=0; i < iter; ++i){
    sign(m, &sk, &s);
  }
  clock_t stop = clock();
  double delta = (double)(stop - start) / CLOCKS_PER_SEC;
  printf("Elapsed time for %d signatures: %f. (%f ms per sig)\n Sign/sec = %f\n", iter, delta, (delta/iter)*1000,iter/delta);

}



int main(){
  srand(time(0));
  //srand(0);
  seed_rng();
  printf("Hello world, signature is Mitaka %u\n", MITAKA_D);
  secret_key sk;
  public_key pk;
  signature s;
  printf("Precomp.\n");
  precomp(&sk);
  //print_sk(&sk);

  load_pk(&pk);



  uint8_t m[32] = {0x46,0xb6,0xc4,0x83,0x3f,0x61,0xfa,0x3e,0xaa,0xe9,0xad,0x4a,0x68,0x8c,0xd9,0x6e,0x22,0x6d,0x93,0x3e,0xde,0xc4,0x64,0x9a,0xb2,0x18,0x45,0x2,0xad,0xf3,0xc,0x61};

  printf("Sign.\n");
  sign(m, &sk, &s);

  printf("Verify: %i\n", verify(m, &pk, &s));

  printf("Speed\n");
  speed();
  run_benchmarks();
  run_tests();



  return 0;
}
