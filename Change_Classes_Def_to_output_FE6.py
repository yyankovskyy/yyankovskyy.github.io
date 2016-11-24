# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 12:17:26 2016

@author: Eugene Yankovsky

Final Exam problem 6
requires to change definition of classes to match the output requirements
"""



# Part 1
class Person(object):     
    def __init__(self, name):         
        self.name = name     
    def say(self, stuff):         
        return self.name + ' says: ' + stuff     
    def __str__(self):         
        return self.name  

class Lecturer(Person):     
    def lecture(self, stuff):         
        return 'I believe that ' + Person.say(self, stuff)  

class Professor(Lecturer): 
    def say(self, stuff): 
        return self.name + ' says: ' + self.lecture(stuff)

class ArrogantProfessor(Person): 
    def say(self, stuff): 
        return self.name + ' says: It is obvious that ' + Person.say(self, stuff)
    def lecture(self, stuff):         
        return 'It is obvious that ' + Person.say(self, stuff)  
        
        
e = Person('eric') 
le = Lecturer('eric') 
pe = Professor('eric') 
ae = ArrogantProfessor('eric')


ae.say('the sky is blue')
# >>> 'eric says: It is obvious that eric says: the sky is blue'

ae.lecture('the sky is blue')
# >>> 'It is obvious that eric says: the sky is blue'

# Part 2
class Person(object):     
    def __init__(self, name):         
        self.name = name     
    def say(self, stuff):         
        return self.name + ' says: ' + stuff     
    def __str__(self):         
        return self.name  

class Lecturer(Person):     
    def lecture(self, stuff):         
        return 'I believe that ' + Person.say(self, stuff)  

class Professor(Lecturer): 
    def say(self, stuff): 
        return self.name + ' says: ' + self.lecture(stuff)

class ArrogantProfessor(Lecturer): 
    def say(self, stuff): 
        return self.name + ' says: It is obvious that I believe that ' + Lecturer.say(self, stuff)
    def lecture(self, stuff):         
        return 'It is obvious that I believe that ' + Lecturer.say(self, stuff)  
        
        
e = Person('eric') 
le = Lecturer('eric') 
pe = Professor('eric') 
ae = ArrogantProfessor('eric')


ae.say('the sky is blue')
# >>> 'eric says: It is obvious that I believe that eric says: the sky is blue'

ae.lecture('the sky is blue')
# >>>'It is obvious that I believe that eric says: the sky is blue'

# Part 3
class Person(object):     
    def __init__(self, name):         
        self.name = name     
    def say(self, stuff):         
        return self.name + ' says: ' + stuff     
    def __str__(self):         
        return self.name  

class Lecturer(Person):     
    def lecture(self, stuff):         
        return 'I believe that ' + Person.say(self, stuff)  

class Professor(Lecturer): 
    def say(self, stuff): 
        return 'Prof. ' + self.name + ' says: ' + self.lecture(stuff)

class ArrogantProfessor(Lecturer): 
    def say(self, stuff): 
        return self.name + ' says: It is obvious that I believe that ' + Lecturer.say(self, stuff)
    def lecture(self, stuff):         
        return 'It is obvious that I believe that ' + Lecturer.say(self, stuff)  

e = Person('eric') 
le = Lecturer('eric') 
pe = Professor('eric') 
ae = ArrogantProfessor('eric')

pe.say('the sky is blue')
#>>> 'Prof. eric says: I believe that eric says: the sky is blue'
