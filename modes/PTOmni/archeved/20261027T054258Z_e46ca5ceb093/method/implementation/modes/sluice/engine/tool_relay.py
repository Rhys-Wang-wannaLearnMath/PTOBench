#!/usr/bin/env python3
"""Unprivileged native stdio transport; only the trusted parent owns the controller."""
import os
import socket
import sys
import threading


def main():
    path = os.environ['SLUICE_TOOL_SOCKET']
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect(path)
        def receive():
            with connection.makefile('rb') as reader:
                while data := reader.read1(65536):
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
        worker = threading.Thread(target=receive)
        worker.start()
        try:
            for line in sys.stdin.buffer:
                connection.sendall(line)
        finally:
            connection.shutdown(socket.SHUT_WR)
            worker.join()


if __name__ == '__main__':
    main()
