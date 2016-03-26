#!/bin/bash
# ASIO_STANDALONE - No Boost
# ASIO_ENABLE_HANDLER_TRACKING - Debugging information
clang++ server.cpp -o server -std=c++11 -pthread \
    -DASIO_STANDALONE \
    -DASIO_ENABLE_HANDLER_TRACKING
