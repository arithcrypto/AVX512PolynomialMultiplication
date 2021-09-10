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

Karat5_5 : Karat5_5.o fonctions.o
	gcc-10 -o Karat5_5 Karat5_5.o fonctions.o $(OPTIONS)

fonctions.o : fonctions.c fonctions.h
	gcc-10 -c fonctions.c $(OPTIONS)

Karat5_5.o : fonctions.o
	gcc-10 -c Karat5_5.c $(OPTIONS)

bench:
	for size in 512 1024 2048 4096 ; do \
		make -B TEST=1 SIZE=$$size ; \
		./Karat5_5 ; \
	done
	
	
		
all : Karat5_5


clean :
	rm *.o -fv *~

cleanall : clean
	rm Karat5_5 -f

