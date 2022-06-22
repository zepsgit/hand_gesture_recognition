class Flag:
  flag = False

f = Flag()
print("Old flag: ")
print(f.flag)

setattr(f, 'flag', True)

print("Flag Now is: ")
print(f.flag)