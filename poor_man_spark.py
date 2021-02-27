#!/usr/bin/env python2.7

# Author: Jarrod Pas <j.pas@usask.ca>

from glob import glob
from itertools import groupby
from multiprocessing import JoinableQueue, Process
import os
import sys
from time import sleep

def mapper(f, queue):
    # Wraps function f which should iterates over items in JoinableQueue
    def wrapper():
        while True:
            item = queue.get()
            f(item)
            queue.task_done()
    return wrapper

def gpu_worker(files):
    sleep(0.25)
    print 'gpu %s %s' % (os.getpid(), files)

def cpu_worker(files):
    sleep(1)
    print 'cpu %s %s' % (os.getpid(), files)

def cpu_worker2(queue):
    while True:
        files = queue.get()
    	sleep(1)
    	print 'cpu2 %s %s' % (os.getpid(), files)
        queue.task_done()

def main(args):
    n_gpu = int(args[0])
    n_cpu = int(args[1])

    # create file glob
    # sorted by mrc689
    files = sorted(glob(args[2] + '/*.tif'))

    # grouping function
    image_id = lambda path: os.path.basename(path).split('_')[1]

    # queue of tasks files
    queue = JoinableQueue()
    for g, group in groupby(files, image_id):
        queue.put(list(group))

    workers = [
        # cpus
        Process(target=mapper(cpu_worker, queue))
        for cpu in range(n_cpu)
    ] + [
        # gpus
        Process(target=mapper(gpu_worker, queue))
        for gpu in range(n_gpu)
    ]

    # start workers
    for worker in workers:
        worker.start()

    try:
        # wait for everything to finish
        queue.join()
    finally:
        # make sure workers get stopped
        for worker in workers:
            worker.terminate()

if __name__ == '__main__':
    exit(main(sys.argv[1:]))


