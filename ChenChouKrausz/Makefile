OPTIONS= -O3 -g -mavx2 -mpclmul -funroll-all-loops 


rkara3_mul : rkara3_mul_avx2.o gf2x_karatsuba.o
	gcc-10 -o rkara3_mul rkara3_mul_avx2.o gf2x_karatsuba.o gf2xmul_AVX2.o gf2x_mul_pclmul.o $(OPTIONS)

gf2x_karatsuba.o : gf2x_karatsuba.c gf2x_karatsuba.h
	gcc-10 -c gf2x_karatsuba.c $(OPTIONS)
	
gf2x_mul_pclmul.o : gf2x_mul_pclmul.c
	gcc-10 -c gf2x_mul_pclmul.c $(OPTIONS)

gf2xmul_AVX2.o : gf2xmul_AVX2.c
	gcc-10 -c gf2xmul_AVX2.c $(OPTIONS)

rkara3_mul_avx2.o : gf2x_karatsuba.o gf2xmul_AVX2.o gf2x_mul_pclmul.o
	gcc-10 -c rkara3_mul_avx2.c $(OPTIONS)

		
all : rkara3_mul


clean :
	rm *.o -fv *~

cleanall : clean
	rm rkara3_mul -f

