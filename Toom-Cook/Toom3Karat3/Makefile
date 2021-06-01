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

ToomCook : ToomCook.o fonctions.o
	gcc -o ToomCook ToomCook.o fonctions.o $(OPTIONS)

fonctions.o : fonctions.c fonctions.h
	gcc -c fonctions.c $(OPTIONS)

ToomCook.o : fonctions.o
	gcc -c ToomCook.c $(OPTIONS)

bench:
	for size in 2048 4096 8192 ; do \
		make -B TEST=1 SIZE=$$size ; \
		./ToomCook ; \
	done
	
	
		
all : ToomCook


clean :
	rm *.o -fv *~

cleanall : clean
	rm ToomCook -f

