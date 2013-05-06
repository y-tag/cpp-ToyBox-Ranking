SRCTOP = ..

include $(SRCTOP)/Makefile.defs

INCLUDES += -I../lib \

LIBS += -L../lib \
        -Wl,-rpath,../lib \
        -lranking \

include $(SRCTOP)/Makefile.rules

all:eval-ranking

eval-ranking:eval_ranking.cc
	$(CXX) -o $@ $^ $(CFLAG) $(INCLUDES) $(LIBS)

clean:
	rm -f *~ *.o eval-ranking

