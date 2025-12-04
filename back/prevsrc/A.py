
import torch as tc
import calculator as calc 
import datetime 
from lib import calculator as libcalc
from lib.libEx import calculator as libcalc2

import numpy as np

import lib.QS as QS


print("....".join(["a", "b", "c"]))

klambda = lambda X: -X if X < 0  else  X

print(klambda(-3))

print(calc.add(3,5))

print(libcalc.add(3,5))

print(libcalc2.add(5,5))

today = datetime.datetime.now()

print(today)



class Animal:
    def __init__(self,name = "Animal"):
        self.name = name
    
    def __del__(self):
        print(f"{self.name} is being destroyed")
        self.name = None

    def __str__(self):
        return f"Animal: {self.name}"

    def speak(self):
        return f"{self.name} makes a noise"
    def MyName(self):
        return self.name

    def Rename(self, newname):
        self.name = newname 

class Dog(Animal):
    def speak(self):
        return f"{self.name} barks"

class Bird(Animal):
    def speak(self):
        return f"{self.name} sings"

animal = Animal("Generic animal")
print(animal.speak())
print(animal.MyName())
print(animal)

dog = Dog("Buddy")
print(dog.speak())
print(dog.MyName())
print(dog)

bird = Bird("Bulue jay")
print(bird.speak())
print(bird.MyName())
print(bird)

class Counter:
  count = 0
  def __init__(self):
    self.count+=1
  def __str__(self):
    return f"{self.count}"
a = Counter()
print(a)

b = Counter()
print(b)

QS.Lines()

