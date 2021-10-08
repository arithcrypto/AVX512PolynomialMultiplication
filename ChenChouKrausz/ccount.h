

#ifndef _CCOUNT_H
#define _CCOUNT_H

//#include "tests.c"

#if defined(__i386__)

static __inline__ unsigned long long rdtsc(void)
{
  unsigned long long int x;
     __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
     return x;
}
#elif defined(__x86_64__)

//typedef unsigned long long int unsigned long long;

static __inline__ unsigned long long rdtsc(void)
{
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

#elif defined(__powerpc__)

typedef unsigned long long int unsigned long long;

static __inline__ unsigned long long rdtsc(void)
{
  unsigned long long int result=0;
  unsigned long int upper, lower,tmp;
  __asm__ volatile(
                "0:                  \n"
                "\tmftbu   %0           \n"
                "\tmftb    %1           \n"
                "\tmftbu   %2           \n"
                "\tcmpw    %2,%0        \n"
                "\tbne     0b         \n"
                : "=r"(upper),"=r"(lower),"=r"(tmp)
                );
  result = upper;
  result = result<<32;
  result = result|lower;

  return(result);
}

#endif
/*
#define STAMP(C){                                  \
 asm volatile("movl %%esi, %[c]"                   \
   : [c] "=m" (C): : "esi");                       \
}
*/
#define STAMP(C){				   \
    C=rdtsc();                                     \
}


#define SAVE_TIME(C){                              \
 asm volatile("movl %%esi, %[c]"                   \
   : [c] "=m" (C): : "esi");                       \
}
#define START(){                                   \
 asm volatile("xorl %%esi, %%esi"                  \
   : : : "esi");                                   \
 asm volatile("rdtsc"                              \
   : : : "eax", "edx");                            \
 asm volatile("subl %%eax, %%esi"                  \
   : : : "eax", "esi");                            \
}
#define STOP(){                                    \
 asm volatile("rdtsc"                              \
   : : : "eax", "edx");                            \
 asm volatile("addl %%eax, %%esi"                  \
   : : : "eax", "esi");                            \
}
#define SERIALIZE(){                               \
 asm volatile("xorl %%eax,%%eax"                   \
   : : : "eax");                                   \
 asm volatile("cpuid"                              \
   : : : "eax", "ebx", "ecx", "edx");              \
}
#define ACCESS_TSC(){                              \
 asm volatile("rdtsc"                              \
   : : : "eax", "edx");                            \
}


#endif

