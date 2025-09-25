def add_numbers(a: int, b: int) -> int:
    return a + b

def greet_user(name: str) -> str:
    return f"Hello, {name}!"

def factorial(n: int) -> int:
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

def is_prime(num: int) -> bool:
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True
