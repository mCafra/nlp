import spacy
  
#load core english library
# nlp = spacy.load("en_core_web_sm")
  
#take unicode string  
#here u stands for unicode
doc = nlp(u"I Love Coding. Geeks for Geeks helped me in this regard very much. I Love Geeks for Geeks.")
#to print sentences
for sent in doc.sents:
  print(sent)