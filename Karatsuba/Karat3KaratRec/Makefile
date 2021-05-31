OPTIONS= -O3 -g -march=tigerlake -lm -lgf2x

ifdef SIZE
        MACROS= -DT_3W=$(SIZE)
else      
        MACROS= -DT_3W=2048  
endif

ifdef TEST
	OPTIONS+= -DTEST
endif

OPTIONS+=$(MACROS)

Karat3 : Karat3.o fonctions.o
	gcc -o Karat3 Karat3.o fonctions.o $(OPTIONS)

fonctions.o : fonctions.c fonctions.h
	gcc -c fonctions.c $(OPTIONS)

Karat3.o : fonctions.o
	gcc -c Karat3.c $(OPTIONS)

bench:
	for size in 512 1024 2048 4096 8192 16384 32768 ; do \
		make -B TEST=1 SIZE=$$size ; \
		./Karat3 ; \
	done
	
	
		
all : Karat3


clean :
	rm *.o -fv *~

cleanall : clean
	rm Karat3 -f

