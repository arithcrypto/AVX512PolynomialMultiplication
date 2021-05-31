OPTIONS= -O3 -g -march=tigerlake -lm -lgf2x

ifdef SIZE
        MACROS= -DT_5W=$(SIZE)
else      
        MACROS= -DT_5W=2048  
endif

ifdef TEST
	OPTIONS+= -DTEST
endif

OPTIONS+=$(MACROS)

Karat5 : Karat5.o fonctions.o
	gcc -o Karat5 Karat5.o fonctions.o $(OPTIONS)

fonctions.o : fonctions.c fonctions.h
	gcc -c fonctions.c $(OPTIONS)

Karat5.o : fonctions.o
	gcc -c Karat5.c $(OPTIONS)

bench:
	for size in 512 1024 2048 4096 8192 16384 ; do \
		make -B TEST=1 SIZE=$$size ; \
		./Karat5 ; \
	done
	
	
		
all : Karat5


clean :
	rm *.o -fv *~

cleanall : clean
	rm Karat5 -f

