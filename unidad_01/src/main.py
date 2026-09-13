import pandas as pd
import numpy as np

def hello_world():
    print()
    print("-" * 60)
    title = "HOLA MUNDO"
    message = "Hello, world! Bienvenido al entorno dev"
    border = "=" * 58

    print(f"+{border}+")
    print(f"| {title.center(56)} |")
    print(f"| {message.center(56)} |")
    print(f"+{border}+")
    print("-" * 60)

print(pd.__version__)
print(np.__version__)
hello_world()