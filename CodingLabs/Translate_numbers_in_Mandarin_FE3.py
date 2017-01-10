# -*- coding: utf-8 -*-
"""
Numbers in Mandarin follow 3 simple rules.

There are words for each of the digits from 0 to 10.
For numbers 11-19, the number is pronounced as "ten digit",
 so for example, 16 would be pronounced (using Mandarin) as "ten six".
For numbers between 20 and 99, the number is pronounced as “digit ten digit”, 
so for example, 37 would be pronounced (using Mandarin) as "three ten seven".
 If the digit is a zero, it is not included.

This procedure procedure converts an American number (between 0 and 99), 
written as a string, into the equivalent Mandarin.

@author: Eugene yankovsky
"""

# Python 2.7        
def convert_to_mandarin(us_num):
    '''
    us_num, a string representing a US number 0 to 99
    returns the string mandarin representation of us_num
    '''
    d = { 0:'ling', 1:'yi', 2:'er', 3:'san', 4: 'si',
          5:'wu', 6:'liu', 7:'qi', 8:'ba', 9:'jiu', 
          10: 'shi', 20: 'er shi',30: 'san shi', 
          40: 'si shi', 50: 'wu shi', 60: 'liu shi',
          70: 'qi shi', 80: 'ba shi', 90: 'jiu shi' }
          
    num=int(filter(str.isdigit, us_num))
    if (num < 10):
        return d[num]

    if (num < 100):
        if num % 10 == 0: return d[num]
        else: return d[num // 10 * 10] +' '+ d[num % 10]
        
# Python 3.3       
def convert_to_mandarin(us_num):
    '''
    us_num, a string representing a US number 0 to 99
    returns the string mandarin representation of us_num
    '''
    d = { 0:'ling', 1:'yi', 2:'er', 3:'san', 4: 'si',
          5:'wu', 6:'liu', 7:'qi', 8:'ba', 9:'jiu', 
          10: 'shi', 20: 'er shi',30: 'san shi', 
          40: 'si shi', 50: 'wu shi', 60: 'liu shi',
          70: 'qi shi', 80: 'ba shi', 90: 'jiu shi' }
    num=int(str(''.join(us_num)))
    if (num < 10):
        return d[num]

    if (num < 100):
        if num % 10 == 0: return d[num]
        else: return d[num // 10 * 10] +' '+ d[num % 10]