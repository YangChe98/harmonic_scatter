FILES =	Makefile main.c  forward.c forward.h 

default: all

all: main

clean:
	-/bin/rm -f *.o *.tmp plot.eps PI[0-9]* \
		*.G.* *.A.* *.Aalpha.* *.Abeta.* \
		*.Gx.* *.Gy.* *.Gz.* *.x.* *.y.* *.z.* *.b.* *.x0.*

veryclean: clean
	-/bin/rm -f poisson simplest maxwell heat mesh*.* eigen elastic \
		maxwell-complex maxwell-eigen maxwell-eigen1 non-smooth \
		navier-stokes main *.vtk

#include /share/soft/phg/phg-0.9.4-mvapich2-20190318/share/phg/Makefile.inc
#include $(PHG_MAKEFILE_INC)
#include /soft/apps/phg/gcc-10.2.0/mvapich2-2.3.5/phg-0.9.6/share/phg/Makefile.inc
#include /soft/apps/phg/intel-2017u4/impi-2017u3/phg-0.9.6/share/phg/Makefile.inc 
include /home/yangche/github/phg-0.9.7/Makefile.inc

main.o: main.c forward.h 

forward.o: forward.c forward.h

main: main.o  forward.o
	${LINKER} ${LDFLAGS} -o $@ $^ ${LIBS} 

.PHONY: default all clean veryclean lib