num1 = int(input("Enter no.of items in the list 1 : "))
list1 = []
for i in range(0,num1):
    item = int(input("Enter item to the list 1 : "))
    list1.append(item)
print(list1)

num2 = int(input("\nEnter no.of items in the list 2 : "))
list2 = []
for i in range(0,num2):
    item = int(input("Enter item to the list 2 : "))
    list2.append(item)
print(list2)

list3 = []
for i in list1:
    if(i in list2):
        list3.append(i)
print("\nList with common elements : ",list3)
        

        
