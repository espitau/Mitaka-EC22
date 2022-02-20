#ifndef PARAM_H
#define PARAM_H



#define MITAKA_D 1024
#define MITAKA_K 320
#define MITAKA_Q 12289
#define MSG_BYTES 32
#define R 1.32
#define R_SQUARE 1.7424

#if MITAKA_D == 512
  #define MITAKA_LOG_D 9
  #define GAMMA_SQUARE 100047795
  #define SIGMA_SQUARE 89985.416
#elif MITAKA_D == 1024
  #define MITAKA_LOG_D 10
  #define SIGMA_SQUARE 116245
  #define GAMMA_SQUARE 258488745.55447942
#endif


#endif