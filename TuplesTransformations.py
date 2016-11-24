# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 12:55:30 2016

@author: anayankovsky

Ch5. Use of tuples:
content is not mutable
"""

def quotent_and_reminder(x,y):
    q = x//y
    r = x%y
    return(q, r)

(quot, rem) = quotent_and_reminder(4,5)

(quot, rem) 


# A function to select min, max, and a number of unique words out of a data sample
def get_date(aTuple):
    nums = ()
    words = ()
    for t in aTuple:
        nums=nums + (t[0],)
        if t[1] not in words:       # to save unique words
            words = words + (t[1],)
    min_nums = min(nums)
    max_nums = max(nums)
    unique_words=len(words)
    return (min_nums, max_nums, unique_words)
    
(small, large, words) = get_date(((1,'mine'),(3,'yours'),(5,'ours'),(7,'mine')))

small = 1
large = 7
words = 3

# Excersises
x = (1, 2, (3,'John',4),'Hi')

type(x[0])
x[0]

type(x[2])
x[2]

type(x[-1])
x[-1]

x[2][2]
type(x[2][2])

x[2][-1]
type(x[2][-1]])

x[-1][-1]
type(x[-1][-1])

x[-1][2]

x[0:1]
type(x[0:1])

len(x)
type(len(x))

2 in x

x[0] = 8
type(x[0] = 8)

#  Takes a tuple as input, and returns a new tuple as output, 
# where every other element of the input tuple is copied, 
# starting with the first one. So if test is the tuple ('I', 'am', 'a', 'test', 'tuple'), then evaluating oddTuples on this input would return the tuple ('I', 'a', 'tuple').
def oddTuples(aTup):
    newTu = ()
    for i in range(0, len(aTup)):
        if i % 2 == 0:
            newTu += (aTup[i],) # make it possible to add a new element to the end of the tuple
            # tupel is not mutable
            # singleton of tuple is tuple(x,), pay attention of the comma 
    return newTu
    
print oddTuples(('I', 'am', 'a', 'test', 'tuple'))