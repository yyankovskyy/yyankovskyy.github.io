"""This code implement a class according to the specifications
 in the template file usresident.py. The file contains a Person class similar 
 to what you have seen in lecture and a USResident class (a subclass of Person).
 Person is already implemented for you and you will have to implement 
 two methods of USResident."""

class USResident(Person):
    """ 
    A Person who resides in the US.
    """
    def __init__(self, name, status):
        """ 
        Initializes a Person object. A USResident object inherits 
        from Person and has one additional attribute:
        status: a string, one of "citizen", "legal_resident", "illegal_resident"
        Raises a ValueError if status is not one of those 3 strings
        """
        Person.__init__(self, name)
        if status != 'citizen' and status != 'legal_resident' and status != 'illegal_resident':
            raise ValueError()
        else:
                self.status = status

    def getStatus(self):
        """
        Returns the status
        """
        return self.status
        
a = USResident('Tim Beaver', 'citizen')
print(a.getStatus())