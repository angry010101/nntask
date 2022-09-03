def program():
    n = int(input("Input: "))
    r = int(n*(n+1)/2) # r=0; for i in range(1, n):  r += i;
    print(f"Output {r}")

if __name__ == '__main__':
    program()

