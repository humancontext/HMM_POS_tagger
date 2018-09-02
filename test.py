from nltk.corpus import *

sents = treebank.tagged_sents(tagset = "")


possible_tags = []
sum = 0
for sent in sents:
   for token in sent:
       sum += 1
        # if token[1] not in possible_tags:
        #     possible_tags.append(token[1])
print(sum)
                


    