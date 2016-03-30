#!/usr/bin/env python3
#
# The Python side of the C++/Python interface
#
# Based on:
# https://docs.python.org/3/howto/sockets.html
# http://stackoverflow.com/a/1716173
# http://stackoverflow.com/a/27345949
import os
import sys
import json
import select
import socket
import argparse
import threading
from time import sleep
from collections import deque
import multiprocessing
from multiprocessing.managers import SyncManager
from learning import readNetworkData, shrinkSamples, GPRParams, RunPath
import matplotlib.pyplot as plt

# How we separate one set of data from the next
delimiter = b'\0'

# Only show one figure, just update it on key press
#plt.ion()
#fig = plt.figure(figsize=(15,8))
#plt.show()

#
# Allow working with a deque between threads
#
SyncManager.register('deque', deque)

#
# Work with data and commands
#
class NetworkData:
    def __init__(self, data, commands):
        self.data = data
        self.commands = commands

    # Add data/commands
    def addData(self, d):
        self.data.append(d)

    def addCommand(self, c):
        self.commands.append(c)

    # Get one and pop off that we've used this data
    def getData(self):
        # Must copy since AutoProxy[deque] doesn't allow indexing
        c = self.data.copy()

        if c:
            d = c[0]
            self.data.popleft()
            return d

        return None

    # Just get *all* the data, so we can just keep on running the thermal
    # identification on the last so many data points
    def getAllData(self):
        return self.data.copy()

    # Get one and pop off that we've sent this command
    def getCommand(self):
        c = self.commands.copy()

        if c:
            d = c[0]
            self.commands.popleft()
            return d

        return None

#
# Show process info
#
def processInfo(title):
    if debug:
        print(title)
        print('  module name:', __name__)
        print('  parent process:', os.getppid())
        print('  process id:', os.getpid())
        print()

#
# Processing thread, where we do thermal identification
#
def processingThread(manager, maxLength):
    processInfo("processingThread")

    while True:
        # Get the last so many data points
        networkData = manager.getAllData()

        # We want quite a few points
        #
        # Note: 375 at 25 Hz is 15 seconds, so wait for at least 15 seconds of
        # data before starting
        if not networkData:
            if debug:
                print("No data yet")
            sleep(1)
            continue

        if len(networkData) < 375:
            if debug:
                print("Only have", len(networkData))
            sleep(1)
            continue

        data = readNetworkData(networkData)

        # Take only every n'th point
        data = shrinkSamples(data, 5)

        # Run GPR
        if debug:
            print("Running GPR")
        RunPath(data, gprParams=GPRParams(
            theta0=1e-2,
            thetaL=1e-10,
            thetaU=1e10,
            nugget=1,
            random_start=10))

    print("Exiting processingThread")

#
# Thread to get data from the network connection and send commands through it
#
def networkingThread(server, port, manager, maxLength):
    processInfo("networkingThread")

    # Connect to server
    print("Connecting to ", server, ":", port, sep="")
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((server, port))
    client.setblocking(1)

    # Store the input/output data since we will be sending/receiving it
    # through multiple recv() and send() calls
    recvBuf = b''
    sendBuf = b''

    # Count how many messages we get, debugging
    i = 0

    while True:
        # See if we're ready to read/write data or if there's been an error
        #inputready, outputready, exceptready = \
        #    select.select([client],[client],[client],300)

        # At the moment, only wait for read ready, so we don't hog the CPU
        # since write is always ready if we aren't busy writing anything...
        inputready, outputready, exceptready = \
            select.select([client],[],[],300)

        # Read data
        for s in inputready:
            # Receive up to 4096 bytes
            data = s.recv(4096)

            # If no data was received, the connection was closed
            if not data:
                s.close()
                print("Exiting, connection closed")
                break

            # Append to already-received data
            recvBuf += data

            # Process received messages
            while True:
                # Split into (before,delim,after)
                before, delimfound, after = recvBuf.partition(delimiter)

                # If we found a delimiter, we have a complete message
                # before that
                if len(delimfound) > 0:
                    receivedData = json.loads(before.decode('utf-8'))
                    manager.addData(receivedData)

                    if debug:
                        i += 1
                        print(i, "Received:", receivedData)

                    # Save what we haven't processed already
                    recvBuf = after

                    # Just add a test command to send every time we receive
                    # new data
                    #command = json.dumps({
                    #    "command":"type",
                    #    "date": "Test Date",
                    #    "lat": 1.0,
                    #    "lon": 2.0,
                    #    "alt": 500.0,
                    #    "radius": 15.0
                    #    })
                    #print("Sending:", command)
                    #sendBuf += command.encode('utf-8') + delimiter

                # If we don't have any messages, continue receiving more
                # data
                else:
                    break

        # Write data
        for s in outputready:
            # If we have data to send, send some of it
            if sendBuf:
                sent = s.send(sendBuf)

                # Remove the data we ended up being able to send
                sendBuf = sendBuf[sent:]

                # If we couldn't send something, then error
                if sent == 0:
                    s.close()
                    print("Exiting, could not send data")
                    break

        # If it's in error, exit
        for s in exceptready:
            s.close()
            print("Exiting, connection closed on error")
            break

    print("Exiting networkingThread")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest='server', type=str, default="127.0.0.1:2050",
            help='address of the C++/Python interface server to connect to')
    parser.add_argument('-d', dest='debug', action='store_true',
            help='debugging information')
    args = parser.parse_args()

    # Get the server and port number from the input arguments
    try:
        server, port = args.server.split(":")
        port = int(port)
    except ValueError:
        print("Error: invalid server address, example: localhost:2050")
        sys.exit(1)

    # Make the debug flag global
    global debug
    debug = args.debug

    processInfo("mainThread")

    # Max length of data to keep
    maxLength = 750

    with SyncManager() as manager:
        # Data to be passed back and forth between processes
        data = manager.deque(maxlen=maxLength)
        commands = manager.deque(maxlen=maxLength)

        # Functions to operate on these deques
        nd = NetworkData(data, commands)

        # Start the processes
        n = multiprocessing.Process(target=networkingThread,args=[server, port, nd, maxLength])
        p = multiprocessing.Process(target=processingThread,args=[nd, maxLength])
        n.start()
        p.start()
        p.join()
        n.join()
