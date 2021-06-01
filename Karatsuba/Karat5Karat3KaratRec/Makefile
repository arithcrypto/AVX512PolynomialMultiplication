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

Karat5_3 : Karat5_3.o fonctions.o
	gcc -o Karat5_3 Karat5_3.o fonctions.o $(OPTIONS)

fonctions.o : fonctions.c fonctions.h
	gcc -c fonctions.c $(OPTIONS)

Karat5_3.o : fonctions.o
	gcc -c Karat5_3.c $(OPTIONS)

bench:
	for size in 512 1024 2048 4096 8192 ; do \
		make -B TEST=1 SIZE=$$size ; \
		./Karat5_3 ; \
	done
	
	
		
all : Karat5_3


clean :
	rm *.o -fv *~

cleanall : clean
	rm Karat5_3 -f

