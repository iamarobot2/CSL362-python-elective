num = int(input("Enter no.of items in the list : "))
integers = []
for i in range(0,num):
    item = int(input("Enter item to the list : "))
    integers.append(item)
print(integers)
number = int(input("Enter a number : "))
li = []
for i in range(0,num):
    if(integers[i] < number):
        li.append(integers[i])
print(li)
    
