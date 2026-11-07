# ygg: AVX-512 VNNI build for the host (Ice Lake, Cascade Lake, Sapphire Rapids or later). The paper's timings use
# ARCH=sapphirerapids; set ARCH for a cross build.
# ygg_simde: portable SIMDe build for hosts without it, same results, not for timing.
CXX ?= g++
FLAGS = -O3 -std=c++17 -ffp-contract=off -Wall
ARCH ?= native
# path to https://github.com/simd-everywhere/simde
SIMDE ?= simde
PYTHON ?= python

ygg: ygg.cpp
	$(CXX) $(FLAGS) -march=$(ARCH) -o $@ $<

ygg_simde: ygg.cpp ygg_simde.h
	$(CXX) $(FLAGS) -march=native -Wno-psabi -DYGG_SIMDE -I$(SIMDE) -o $@ $<

test: ygg
	cd .. && YGG=cpu/ygg $(PYTHON) -m unittest cpu/tests/test_kernel.py

clean:
	rm -f ygg ygg_simde
.PHONY: test clean
