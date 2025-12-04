class Counter:
    count = 0
    def __init__(self):
        Counter.count += 1
        self.count = Counter.count
    def __str__(self):
        return f"{self.count}"

a = Counter()
print(a)

b = Counter()
print(b)

c = Counter()
print(c)


class MathUtil:
    # 정적 메서드 정의
    @staticmethod
    def add(x, y):
        """두 수를 더해서 반환"""
        return x + y
    
    @staticmethod
    def multiply(x, y):
        """두 수를 곱해서 반환"""
        return x * y
    
    
    def is_even(n):
        """짝수 여부 판별"""
        return n % 2 == 0

# 클래스명으로 호출 (권장 방식)
print(MathUtil.add(3, 5))         # 8
print(MathUtil.multiply(4, 6))    # 24
print(MathUtil.is_even(10))       # True

# 인스턴스로 호출도 가능하지만 권장 X
m = MathUtil()
print(m.add(7, 2))                # 9