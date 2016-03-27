#!/usr/bin/env python3
#
# The Python side of the C++/Python interface
#
# Based on:
# https://docs.python.org/3/howto/sockets.html
# http://stackoverflow.com/a/1716173

import sys
import json
import select
import socket
import argparse
from time import sleep

# How we separate one set of data from the next
delimiter = b'\0'

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

    try:
        # Connect to server
        print("Connecting to ", server, ":", port, sep="")
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((server, port))
        client.setblocking(0)

        # Store the input/output data since we will be sending/receiving it
        # through multiple recv() and send() calls
        recvBuf = b''
        sendBuf = b''

        while True:
            # See if we're ready to read/write data or if there's been an error
            inputready, outputready, exceptready = \
                select.select([client],[client],[client],300)

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
                        print("Received:", receivedData)

                        # Save what we haven't processed already
                        recvBuf = after

                        # Just add a test command to send every time we receive
                        # new data
                        command = json.dumps({
                            "command":"type",
                            "date": "Test Date",
                            "lat": 1.0,
                            "lon": 2.0,
                            "alt": 500.0,
                            "radius": 15.0
                            })
                        print("Sending:", command)
                        sendBuf += command.encode('utf-8') + delimiter

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

    except KeyboardInterrupt:
        print()
