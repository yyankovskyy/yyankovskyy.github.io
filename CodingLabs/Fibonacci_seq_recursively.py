# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 23:57:02 2016

@author: anayankovsky
Fibonacci sequence
"""
def fib(x):
    """assumes x an int>=0
        returns Fibonacci of x """
    if x==0 or x==1:            # base cases
        return 1
    else:
        return fib(x-1) + fib(x-2)
        
