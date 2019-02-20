import random

N = 10
train_samples = 1000
test_samples = 150

f = open("Train.txt", "w")

for i in xrange(train_samples):
    a = []
    sum = 0.0
    count = 0
    for j in xrange(N):
        temp = random.random()
        if temp <= 0.5:
            a.append( random.randint(0,N) )
            sum += a[-1]
            count += 1
        else:
            a.append(0)
    
    if count != 0:
        for item in a:
            print >> f, item,
        print >> f, sum/count

f.close()

f = open("Test.txt", "w")

for i in xrange(test_samples):
    a = []
    sum = 0.0
    count = 0
    for j in xrange(N):
        temp = random.random()
        if temp <= 0.5:
            a.append( random.randint(0,N) )
            sum += a[-1]
            count += 1
        else:
            a.append(0)
    
    if count != 0:
        for item in a:
            print >> f, item,
        print >> f, sum/count

f.close()