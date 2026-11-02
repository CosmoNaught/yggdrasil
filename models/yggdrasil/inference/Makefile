# ygg: AVX-512 VNNI build. ygg_simde: portable SIMDe build, same results, not for timing.
CXX ?= g++
FLAGS = -O3 -std=c++17 -ffp-contract=off -Wall
SIMDE ?= simde  # path to https://github.com/simd-everywhere/simde

ygg: ygg.cpp
	$(CXX) $(FLAGS) -march=sapphirerapids -o $@ $<

ygg_simde: ygg.cpp ygg_simde.h
	$(CXX) $(FLAGS) -march=native -Wno-psabi -DYGG_SIMDE -I$(SIMDE) -o $@ $<

clean:
	rm -f ygg ygg_simde
.PHONY: clean
