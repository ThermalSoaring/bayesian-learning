OUT        = server
SRC        = ${wildcard *.cpp}
OBJ        = ${SRC:.cpp=.o}
DEPENDS    = .depends

CXX        = clang++

# ASIO_ENABLE_HANDLER_TRACKING - Debugging information
CXXFLAGS  += -g -O2 -Wall \
			 #-DASIO_ENABLE_HANDLER_TRACKING
LDFLAGS   += -lboost_system -lboost_thread

all: ${OUT}

${OUT}: ${OBJ}
	${CXX} -o $@ ${OBJ} ${LDFLAGS}

.cpp.o:
	${CXX} -c -o $@ $< ${CXXFLAGS}

${DEPENDS}: ${SRC}
	${RM} -f ./${DEPENDS}
	${CXX} ${CXXFLAGS} -MM $^ >> ./${DEPENDS}

clean:
	${RM} ${OUT} ${OBJ} ${DEPENDS}

-include ${DEPENDS}
.PHONY: all clean
