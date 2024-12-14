# Copyright (c) 2024 Kulagin Aleksandr
CXX = g++

CXXFLAGS = -Wall -Wextra -g -O3

TARGET = main

SRCS = main.cpp gelu_ocl.cpp

OBJS = $(SRCS:.cpp=.o)

all: $(TARGET) run clean

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) -lOpenCL

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ -lOpenCL

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)
