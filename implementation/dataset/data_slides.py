import random

output_nums = 10000
rand_var = random.sample(range(1, 1500000), output_nums)
filename = str(output_nums) + '_new_train.csv'
rand_var.sort()
# print rand_var
j = 0



with open(filename, 'aw+') as outputfile, open('data.csv', 'r') as inputfile:
        for i, line in enumerate(inputfile):
            if i == rand_var[j]:
                # print i, rand_var[j]
                outputfile.write(line)
                j  = j + 1
            if j >= output_nums:
                break



