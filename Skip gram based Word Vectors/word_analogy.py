import os
import pickle
import numpy as np


model_path = './models/'
# loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'r'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

def cosine_similarity(A,B):
	# this function calculates cosine similarity between A and B
    numerator = A.T.dot(B)
    denominator = np.linalg.norm(A)*np.linalg.norm(B)
    return 1.0*(numerator/denominator)


with open("word_analogy_dev.txt") as f:
    line = f.readline()
    
    # arr to store final string results
    output_string_arr = []
    
    # iterating over each line in the file
    while line:
        diff_examples = []
        diff_choices = []
        ex, ch = line.strip().split("||") # split line in examples and choices
        ex = ex.split(",") # examples
        ch = ch.split(",") # choices

        # getting example embedding differences
        for example in ex:
            # getting rid of ""
            example = example[1:-1].split(":") # split over pairs
            diff = embeddings[dictionary[example[0]]] - embeddings[dictionary[example[1]]]
            diff_examples.append(np.array(diff))

        # getting choices embedding differences
        for choice in ch:
            # getting rid of ""
            choice = choice[1:-1].split(":") # split over pairs
            diff = embeddings[dictionary[choice[0]]] - embeddings[dictionary[choice[1]]]
            diff_choices.append(np.array(diff))

        # calculating cosine similarity and averaging
        cs = cosine_similarity(np.array(diff_examples).T, np.array(diff_choices).T)    
        
        cs2 = [0 for _ in range(len(cs[0]))]
        for col in range(len(cs[0])):
            cs2[col] = 1.0*np.sum(cs[:,col])/len(cs)
        
        smallest = min(cs2)
        largest = max(cs2)
        
        # calculating least and most illustrative word
        least_illustrative = ch[cs2.index(smallest)]
        most_illustrative = ch[cs2.index(largest)]

        # construct output string to write to file later
        output_string = ""
        for i in ch:
            output_string += i
            output_string += " "
        output_string = output_string + least_illustrative + " " + most_illustrative     
        output_string_arr.append(output_string)        

        # read next line from txt file
        line = f.readline()


# writing to output file

if loss_model == 'cross_entropy':
    output_file = "temp_dev_ce.txt"
    # output_file = "word_analogy_test_predictions_cross_entropy.txt"
elif loss_model == "nce":
    output_file = "temp_dev_nce.txt"
    # output_file = "word_analogy_test_predictions_nce.txt"

f2 = open(output_file, "w")
for line in output_string_arr:
	f2.write(line)
	f2.write("\n")
f2.close()
'''


#### Uncomment this part to get the top 20 similar words ####
# Code to find top 20 similar words

words = ['first','american','would']

# creating temp dict to store top 20
d = {}

for word in words:
    index = dictionary[word]
    word_emb = embeddings[index]
    
    cs_arr = []

    d[word] = []

    # for each word in dictionary, calculate cosine similarity with current word

    for k,v in dictionary.items():
        # if word is the target word itself, skip it
        if k == word:
            continue

        # get embedding of k
        k_emb = embeddings[v]
        
        # calculate cosine similarity between k_emb and word_emb
        cs = cosine_similarity(k_emb, word_emb)
        cs_arr.append((cs, k))

    # sorting the list in desc order based on cosine similarity
    cs_arr.sort(key=lambda x: x[0], reverse=True)
    top_20 = cs_arr[:20]

    for i in top_20:
        d[word].append(i[1])

print d
'''