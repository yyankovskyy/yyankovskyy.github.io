# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 23:02:18 2016

@author: eugene yankovsky

Functions to analyze song lyrics
"""

# Count the frequency of the words
def lyrics_to_frequencies(lyrics):
    myDict={}
    for word in lyrics:
        if word in myDict:
            myDict[word] +=1
        else:
            myDict[word] = 1
    return myDict

# Ex
she_loves_you = ['she','loves','you','yeah', 'yeah','yeah']
beatles = lyrics_to_frequencies(she_loves_you)

# Define most common words
def most_common_words(freqs):
    values = freqs.values()
    best = max(values)
    words = []
    for k in freqs:
        if freqs[k] ==best:
            words.append(k)
    return (words, best)
    
# Ex 
(common_word, frequency) = most_common_words(beatles)
common_word
frequency

# Find words that happens most frequent
def words_often(freqs, minTimes):
    result = []
    done = False               # Flag
    while not done:
        temp = most_common_words(freqs)
        if temp[1]>=minTimes:
            result.append(temp)
            for w in temp[0]:
                del(freqs[w])
        else:
            done = True
    return result

print(words_often(beatles,2))