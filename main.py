# How NLP works
# Segmentation
# - Split sentences into single line from full-stops and commas
# Tokenising
# - Split sentences into words
# Stop words
#  - Remove stop words for easiest understanding, stop words make words more
#     cohesive
# Stemming
# - Prefix and suffix words with same meaning word (play, plays, playing etc.)
# Lemmatisation
# - Base words for gender/mood
# Speech tagging
# - Recognise noun, verb, adjective, preposition
# Named entity tagging
# - Make NLP understand famous names, countries, movies etc. so they are not 
#   picked up
# Apply NLP algorithms
# - Frequency of words etc.
# - Maive bayes algorithm (?)
# - Word embedding (words with the same context as each other, e.g man, 
#   woman, boy, girl)


from ast import Break
from multiprocessing.resource_sharer import stop
from operator import le
import requests 
import json
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from colorama import Fore, Style

nltk.download('wordnet')
nltk.download('omw-1.4')
# word = 'hello'
# response = requests.get('https://api.dictionaryapi.dev/api/v2/entries/en/'+word)
# parsed = json.loads(response.content)
# print(json.dumps(parsed, indent=4))

DELIMITERS = ['.',',','/',';',':','?']
STOP_WORDS = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]
sampleInput = 'A grasshopper spent the summer hopping about in the sun and singing to his heart\'s content. One day, an ant went hurrying by, looking very hot and weary. "Why are you working on such a lovely day?" said the grasshopper. "I\'m collecting food for the winter," said the ant, "and I suggest you do the same." And off she went, helping the other ants to carry food to their store. The grasshopper carried on hopping and singing. When winter came the ground was covered with snow. The grasshopper had no food and was hungry. So he went to the ants and asked for food. "What did you do all summer when we were working to collect our food?" said one of the ants.'


def replaceForSegmentation(userInput):
    for x in DELIMITERS:
        userInput = userInput.replace(x, x+'&')
    return userInput

def replaceForTokenisation(userInput):
    for x in DELIMITERS:
        userInput = userInput.replace(x, '')
    userInput = userInput.replace('"', '')
    return userInput.lstrip()

def tokenise(userInput):
    index = 1
    TokenisedInput = userInput.split()
    for x in TokenisedInput:
        out = replaceForTokenisation(x)
        print(str(index) + ': ' + formatInput(out,1))
        index += 1 
    input()
    
def segmentation(userInput):
    index = 1
    userInput = replaceForSegmentation(userInput)
    SegmentedInput = userInput.split('&')
    for x in SegmentedInput:
        out = x.replace('&', ' ').lstrip()
        
        print(str(index) + ': ' + formatInput(out,1))
        index += 1 
    input()

def stopWords(userInput, mode):
    index = 1
    if mode == 0:
        TokenisedInput = userInput.split()
        for x in TokenisedInput:
            stopWord = False
            out = replaceForTokenisation(x)
            for y in STOP_WORDS:
                if formatInput(out,1) == y: 
                    stopWord = True
                    break
            if stopWord: 
                print(Fore.RED + str(index) + ': ' + formatInput(out,1))
                Style.RESET_ALL    
            else: 
                Style.RESET_ALL
                print(Fore.WHITE + str(index) + ': ' + formatInput(out,1))
            index += 1 

    elif mode == 1:
        userInput = replaceForSegmentation(userInput)
        SegmentedInput = userInput.split('&')
        for x in SegmentedInput:
            out = x
            words = x.split() 
            for y in words:
                for z in STOP_WORDS:
                    if y == z: 
                        out = out.replace(' ' + y + ' ', ' ')
            print(out.lstrip())

            #     if formatInput(out,1) == y: 
            #         stopWord = True
            #         break
            # print(str(index) + ': ' + formatInput(out,1))
            # index += 1 
    input()
    
def stemming(userInput):
    ps = PorterStemmer()
    index = 1
    TokenisedInput = userInput.split()
    for x in TokenisedInput:
        out = replaceForTokenisation(x)
        print(str(index) + " - " + out + " : ", ps.stem(out))
        index += 1 

def lemmatisation(userInput):
    l = WordNetLemmatizer()
    index = 1
    TokenisedInput = userInput.split()
    for x in TokenisedInput:
        out = replaceForTokenisation(x)
        print(str(index) + " - " + out + " : ", l.lemmatize(out))
        index += 1 


def menu():
    print('=======================')
    print('1. Segmentation - Break input down into sentences')
    print('2. Tokenisation - Break input down into words')
    print('3. Identify Stop words - Highlight stop words, stop words make words more cohesive')
    print('4. Remove Stop words - Remove stop words for easiest understanding, stop words make words more cohesive')
    print('5. Stemming - Prefix and suffix words with same meaning word (play, plays, playing etc.)')
    print('6. Lemmatisation - Base words for gender/mood')
    print('7. Speech tagging - Recognise noun, verb, adjective, preposition')
    print('8. Named entity tagging - Make NLP understand famous names, countries, movies etc. so they are not picked up')
    print('9. Exit')
    print('=======================')
    a = input(": ")
    return int(a)

def formatInput(a, option):
    if option == 0:
        return a.replace('?"', '"?').replace('."','".').replace(',"','",')
    else:
        return a.replace('"?','?"').replace('".', '."').replace('",', ',"')

running = True 
while running:
    # userInput = input("Input Sentence: \n")
    userInput = formatInput(sampleInput,0)
   
    menuOption = menu()
    
    if menuOption == 1: #Segmentation
        segmentation(userInput)
    elif menuOption == 2: #Tokenisation
        tokenise(userInput)
    elif menuOption == 3:
        stopWords(userInput, 0)
    elif menuOption == 4: 
        stopWords(userInput, 1)
    elif menuOption == 5: 
        stemming(userInput)
    elif menuOption == 6:
        lemmatisation(userInput)
    elif menuOption == 9: 
        running = False
    else:
        print("Err with input")