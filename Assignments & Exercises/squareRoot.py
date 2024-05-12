import math
x = float(input("Enter a positive number : "))
tolerence = 0.0000001
estimate = 1.0
while True:
    estimate = (estimate + x / estimate)/2
    difference = abs(x - estimate **2)
    if difference < tolerence:
        print ("The square root of",x,"is approximately",estimate)
        break
print("The  square root of {} is {:.1f}".format(x, estimate))
