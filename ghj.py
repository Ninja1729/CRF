arr = [1,2,3,4,5]


c = 0
tot = len(arr)
for i in arr:
    c += 1
    if tot == c:
        print("yes")
    else:
        print("no")
