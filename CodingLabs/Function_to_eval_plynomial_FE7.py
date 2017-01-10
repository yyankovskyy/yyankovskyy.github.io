# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:38:46 2016

@author: Eugene Yankovsky

Final exam problem 7:
Write a function called general_poly, that meets the specifications below.

For example, general_poly([1, 2, 3, 4])(10) should evaluate to 1234 because 
1∗10^3+2∗10^2+3∗10^1+4∗10^0

So in the example the function only takes one argument with
general_poly([1, 2, 3, 4]) 
and it returns a function that you can apply to a value, 
in this case x = 10 with general_poly([1, 2, 3, 4])(10).
"""


def general_poly (L):
    """ L, a list of numbers (n0, n1, n2, ... nk)
    Returns a function, which when applied to a value x, returns the value
    n0 * x^k + n1 * x^(k-1) + ... nk * x^0 """
    def inner(x):
        x=x
        def function_generator(L, x):
            k = len(L) - 1
            sum = 0
            for number in L:
                sum += number * x ** k   
                k -= 1
            return sum
        
        return function_generator(L, x)

    return inner
    
general_poly([1, 2, 3, 4])(10)
    