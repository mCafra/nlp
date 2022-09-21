
from nltk.stem import WordNetLemmatizer, PorterStemmer


def stemming(userInput):
    ps = PorterStemmer()
    print( userInput + " : ", ps.stem(userInput))
        


while True:
    userInput = input()
    stemming(userInput)