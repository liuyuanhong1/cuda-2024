# Copyright (c) 2024 Kulagin Aleksandr
CXX = g++

CXXFLAGS = -Wall -fopenmp -lpthread -g -O3

TARGET = main

SRCS = main.cpp naive_gemm_omp.cpp

OBJS = $(SRCS:.cpp=.o)

all: $(TARGET) run clean

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)
