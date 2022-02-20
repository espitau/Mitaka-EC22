#include "param.h"

extern double precomp_b10[MITAKA_D];
extern double precomp_b11[MITAKA_D];
extern double precomp_b20[MITAKA_D];
extern double precomp_b21[MITAKA_D];
extern double precomp_h[MITAKA_D];


void load_B(secret_key* sk);
void precomp_GSO(secret_key* sk);
void precomp_sigma(secret_key* sk);
void precomp_beta_hat(secret_key* sk);
void precomp(secret_key* sk);
void load_pk(public_key* pk);
