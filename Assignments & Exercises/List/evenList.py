num = int(input("Enter no.of items in the list : "))
integers = []
for i in range(0,num):
    item = int(input("Enter item to the list : "))
    integers.append(item)
print(integers)
even = []
for i in range(0,num):
    if(integers[i]%2 == 0):
        even.append(integers[i])
print(even)
