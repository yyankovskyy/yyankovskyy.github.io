# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 11:45:17 2016

@author: eugene yankovsky

Ch.5. Dictionaries
"""

names = ['Ana', 'John', 'Denise', 'Katy']
grade = ['B', 'A+', 'A', 'A']
course = [2.00, 6.0001, 20.002, 9.01]

def get_grade(student, name_list, grade_list, course_list):
    i = name_list.index(student)
    grade = grade_list[i]
    course = course_list[i]
    return(course, grade)
    

# Ex
grades = {'Ana':'B','John':'A+','Denise':'A','Katy':'A'}

grades

# Check exisiting grade
grades['John']

#  Add new observation

grades['Sylvan'] = 'A'
grades['Sylvan']

# Check content 
'John' in grades            # True
'Daniel' in grades          # False

del(grades['John'])

# To get a list of keys
grades.keys()

# To get the values
grades.values()

# Exercises
animals = {'a': 'aardvark', 'b': 'baboon', 'c': 'coati'}
animals['d'] = 'donkey'

animals
animals['c']
len(animals)

animals['a'] = 'anteater'
animals['a']

len(animals['a'])

'baboon' in animals

'donkey' in animals.values()

'b' in animals

animals.keys()

del animals['b']
len(animals)


# Ex

animals = { 'a': ['aardvark'], 'b': ['baboon'], 'c': ['coati']}
animals['d'] = ['donkey']
animals['d'].append('dog')




