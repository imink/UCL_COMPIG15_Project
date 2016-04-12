import random

rand_var = random.sample(range(1, 150000), 5000)

rand_var.sort()
print rand_var
j = 0


with open('5000_train.csv', 'aw+') as outputfile, open('data.csv', 'r') as inputfile:
        for i, line in enumerate(inputfile):
            if i == rand_var[j]:
                # print i, rand_var[j]
                outputfile.write(line)
                j  = j + 1
            if j >= 5000:
                break



