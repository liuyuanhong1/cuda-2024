# Copyright (c) 2024 Kulagin Aleksandr
CXX = nvcc

CXXFLAGS = -O3

TARGET = main

SRCS = main.cpp naive_gemm_cuda.cu

OBJS1 = $(SRCS:.cpp=.o)
OBJS = $(OBJS1:.cu=.o)

all: $(TARGET) run clean

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)
