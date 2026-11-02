# ygg: the AVX-512 VNNI kernel (build on the Sapphire Rapids nodes, or anywhere g++ knows the target).
# ygg_simde: portable SIMDe build (exact integer / IEEE semantics, for correctness on hosts without AVX-512; not for timing).
CXX ?= g++
FLAGS = -O3 -std=c++17 -ffp-contract=off -Wall
SIMDE ?= simde  # a checkout of https://github.com/simd-everywhere/simde

ygg: ygg.cpp
	$(CXX) $(FLAGS) -march=sapphirerapids -o $@ $<

ygg_simde: ygg.cpp ygg_simde.h
	$(CXX) $(FLAGS) -march=native -Wno-psabi -DYGG_SIMDE -I$(SIMDE) -o $@ $<

clean:
	rm -f ygg ygg_simde
.PHONY: clean
