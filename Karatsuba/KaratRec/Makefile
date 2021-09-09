OPTIONS= -O4 -g -march=tigerlake -lm -lgf2x
#-DDGK# -DKARAT256 -funroll-all-loops

ifdef SIZE
        MACROS= -DSIZE_N=$(SIZE)
else      
        MACROS= -DSIZE_N=1024  
endif

ifdef TEST
	OPTIONS+= -DTEST
endif

ifdef DGK
        MACROS+= -DDGK
else ifdef DGK2      
        MACROS+= -DDGK2
else ifdef AVX2      
        MACROS+= -DAVX2
else ifdef KARAT256      
        MACROS+= -DKARAT256 
else ifdef SB256      
        MACROS+= -DSB256 
else
	MACROS+= -DKARAT512
endif

OPTIONS+=$(MACROS)

KaratRec : KaratRec.o fonctions.o
	gcc -o KaratRec KaratRec.o fonctions.o $(OPTIONS)

fonctions.o : fonctions.c fonctions.h
	gcc -c fonctions.c $(OPTIONS)

KaratRec.o : fonctions.o
	gcc -c KaratRec.c $(OPTIONS)

bench:  KaratRec
	for size in 1024 2048 4096 8192 16384 32768 65536 131072 ; do \
		make -B TEST=1 SIZE=$$size ; \
		./KaratRec ; \
	done


allbench:
	for mult in DGK DGK2 AVX2 KARAT256 SB256 KARAT512 ; do \
		make bench $$mult=1; \
	done


all : KaratRec


clean :
	rm *.o -fv *~

cleanall : clean
	rm KaratRec -f

