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

Karat3_5 : Karat3_5.o fonctions.o
	gcc -o Karat3_5 Karat3_5.o fonctions.o $(OPTIONS)

fonctions.o : fonctions.c fonctions.h
	gcc -c fonctions.c $(OPTIONS)

Karat3_5.o : fonctions.o
	gcc -c Karat3_5.c $(OPTIONS)

bench:
	for size in 512 1024 2048 4096 8192 ; do \
		make -B TEST=1 SIZE=$$size ; \
		./Karat3_5 ; \
	done
	
	
		
all : Karat3_5


clean :
	rm *.o -fv *~

cleanall : clean
	rm Karat3_5 -f

