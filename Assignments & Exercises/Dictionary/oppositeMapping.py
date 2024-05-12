dictionary = {'k1':'v1','k2':'v2','k3':'v3'}
print(dictionary)
swap_dictionary = {k:v for v,k in dictionary.items()}
print("Dictionary with opposite mapping : ",swap_dictionary)
