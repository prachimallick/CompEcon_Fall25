import calculator

def hypotenuse(a, b):
    """Return the hypotenuse length using calculator.py functions"""
    return calculator.sqrt(
        calculator.add(
            calculator.multiply(a, a),
            calculator.multiply(b, b)
        )
    )

# Test it
print(hypotenuse(3, 4)) 

