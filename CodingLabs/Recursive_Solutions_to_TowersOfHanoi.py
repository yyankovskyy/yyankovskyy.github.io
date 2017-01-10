# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 23:27:45 2016

@author: Eugene Yankovsky

Recursive Solution to the Towers of Hanoi problem
"""

def printMove(fr, to):
    print('move from ' + str(fr)+' to ' + str(to))
    
def Towers(n, fr, to, spare):
    if n==1:
        printMove(fr,to)
    else:
        Towers(n-1,fr,spare,to)     # smaller 
        Towers(1,fr,to,spare)       # build solution to a basic
        Towers(n-1,spare,to,fr)

print(Towers(4,'P1','P2','P3'))



