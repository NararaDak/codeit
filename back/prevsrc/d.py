
class MyClass:
    myValue: str = "Hello class"
    def __init__(self):
        self.myValue1 = "Hello Instance"
        pass
    
   
    @classmethod
    def printClassVal(cls):
        print(cls.myValue)
    
    def printInstanceVal(self):
        print(self.myValue1)


obj1 = MyClass()


MyClass.myValue = "Changed Class World"
MyClass.myValue1 = "Changed Instance World"

obj1.printClassVal()
obj1.printInstanceVal()

