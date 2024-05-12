# 1) Find the largest and smallest number in a list.
# 2) Find the third largest number in a list
# 3) write a python program to take a list list of integers.
#Create another list with those numbers in the master list that are less than a
#number entered by the user.Print the new list contents.
# 4) Write a python program to take a list of integers.create another list to
#store all the even numbers from the master list and print the new list contents
#in ascending order.
# 5) write a python program to take two list of integers and return a list
#containing the elements that are common in both list. Also print the contents
#of the new list.

# 1) :
n1 = int(input("Enter the no. of. elements in the list : "));
list1 = []
list2 = []
for i in range(0,n1):
   element = int(input(f"Enter element {i+1} : "));
   list1.append(element)
print(list1);

n2 = int(input("Enter the no. of. elements in the list : "));
for i in range(0,n2):
   element = int(input(f"Enter element {i+1} : "));
   list2.append(element)
print(list2);
'''print(f"Minimum number in the list : {min(list)}");
print(f"Maximum number in the list : {max(list)}");

print("\nQuestion 2\n");
print(f"Third largest number in the list : {list.sort(reverse=True)}")'''

list3 = []
for i in list1:
   if(i in list2):
      list3.append(i)
print(list3)
      

