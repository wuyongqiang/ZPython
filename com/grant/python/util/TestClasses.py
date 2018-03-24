'''
Created on Jan 24, 2018

@author: grant
'''
class Dog:

    def __init__(self, name):
        self.name = name;
        self.tricks = [];  

    def add_trick(self, trick):
        self.tricks.append(trick)
        
    def get_tricks(self):
        return self.tricks;
    
    def get_name(self):
        return self.name;
    

class Cat:

    def __init__(self, name):
        self.name = name;
        self.tricks = [];  

    def add_trick(self, trick):
        self.tricks.append(trick)
        
    def get_tricks(self):
        return self.tricks;
    
    def get_name(self):
        return self.name;    
    
def catchADog():
    return Dog('wolf');    