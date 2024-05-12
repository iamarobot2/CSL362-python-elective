d1 = {1:'v1', 2:'v2', 3:'v3', 4:'v4'}
d2 = {2:'v5', 4:'v6', 5:'v7', 6:'v8'}
print("\nD1 = ",d1)
print("D2 = ",d2)

print("\nCommon keys : ")
for i in d1:
    if(i in d2):
        print(i)
