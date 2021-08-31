import os
import csv


def loadRcdsFromFile(trainFile):
    if not os.path.exists(trainFile):
        return []
    return list(map(lambda x: x.strip().split("\t"), open(trainFile, 'r', encoding='UTF-8')))


splitReg = r',(?! )'


def getstuff(filename):
    with open(filename, "r", encoding='utf-8') as csvfile:
        datareader = csv.reader(csvfile)
        # yield next(datareader)
        next(datareader)
        for row in datareader:
            yield row


# for i in getstuff('./data/en-fr.csv'):
#     print(i)

# i = getstuff('./data/en-fr.csv')
# print(next(i))
# print(next(i))
# print(next(i))
# print(next(i))
