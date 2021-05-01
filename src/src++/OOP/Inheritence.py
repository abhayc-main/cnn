import numpy as np
import random

class human():
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def speak(self):
        print("DEpends on who I am!!!!!")

    def powerup(self):
        print("DEpends on who I am!!!!!")
    

class firehumans(human):
    def __init__(self, name, age, race):
        super().__init__(name, age)
        self.race = race


    def speak(self):
        print(f"I am {self.name} and i am {self.age} and i am {self.race}")

    def powerup(self):
        print(f"{self.name} has FIRE POWERS!!!!")

rando = human("von",45)
Rick = firehumans("Rick", 5002, "Black")
Rick.speak()
rando.speak()
Rick.powerup()
