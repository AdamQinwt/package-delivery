#!/usr/bin/env python3
# coding=utf-8

# Author: Junjie Wang
# Mail: dreamboy.gns@sjtu.edu.cn

# Website:http://www.dbgns.com
# Blog:http://www.dbgns.com/blog
 
# Created Time: 2019-05-30 22:19:54

import multiprocessing 
import time

x = [1]

def worker():
    for i in range(10, 20):
        x.append(i)

job = multiprocessing.Process(target=worker)
job.start()
job.join()

print(x)
