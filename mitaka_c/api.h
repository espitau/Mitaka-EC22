#ifndef API_H
#define API_H

#include <stdint.h>
#include "poly.h"

void sign(const uint8_t* m, const secret_key* sk, signature* s);
int verify(uint8_t* m, public_key* pk, signature* s);
void sampler(const secret_key* sk, const poly* c1, const poly* c2, poly* v0, poly* v1);

/* Constant-time macros */
#define LSBMASK(c)      (-((c)&1))
#define CMUX(x,y,c)     (((x)&(LSBMASK(c)))^((y)&(~LSBMASK(c))))
#define CFLIP(x,c)      CMUX(x,-(x),c)
#define CZERO64(x)      ((~(x)&((x)-1))>>63)

#endif
