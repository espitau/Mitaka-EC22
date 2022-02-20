#include "randombytes.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
prng p;

int randombytes(uint8_t *buf, size_t n){
  Zf(prng_get_bytes)(&p, buf, n);
  return -1;
}


uint64_t get64(){
  return prng_get_u64(&p);
} 

uint8_t get8(){
  return prng_get_u8(&p);
}


void seed_rng(){
  Zf(prng_init)(&p);
}

