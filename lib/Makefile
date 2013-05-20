SRCTOP = ..

include $(SRCTOP)/Makefile.defs

CXXSRC = ranksvm.cc \
		 ranknet.cc \
		 lambdarank.cc \
		 regression_tree.cc \
		 lambdamart.cc \
		 listnet.cc \
		 listmle.cc \
		 svmlight_reader.cc \

CXXFLAG += -fPIC -Wall

include $(SRCTOP)/Makefile.rules

lib:$(CXXOBJ)
	$(CXX) -shared -o libranking.so $^ $(CXXFLAG)

clean:
	rm -f *~ *.o *.out *.so
