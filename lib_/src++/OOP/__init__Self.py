# The Self keyword just means that we can pass it in for a object.
#The __init__ BASICALLY IS JUST A CONSTRCUTOR KEYWORD.
# For Example...
class Human:
    def __init__(self, name):
        # Here we are writing a function that names a human every time we make new one.
        self.name = name
        print(name)
        # This just means it takes the parameter that we have it when calling the function and does the "parameter ".name = name

    def move(self, name):
        print(name+" MOved!")


h1 = Human("Tom")
h1.move("Tom")
