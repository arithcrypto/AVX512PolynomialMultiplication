

#ifndef _CCOUNT_H
#define _CCOUNT_H


static __inline__ unsigned long long rdtsc(void)
{
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

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

