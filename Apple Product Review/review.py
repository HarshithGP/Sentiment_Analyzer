from collections import Counter
import csv
import re

# Read in the training data.
with open("train.csv", 'r') as file:
  reviews = list(csv.reader(file))

"""  
print(reviews)
for r in reviews:
    print(r)
"""

def get_text(reviews, score):

    text = [r[1].lower() for r in reviews if r[0] == str(score)]
    return text
  # Join together the text in the reviews for a particular tone.
  # We lowercase to avoid "Not" and "not" being seen as different words, for example.
  # return " ".join([r[1].lower() for r in reviews if r[0] == str(score)])

positive_text = get_text(reviews, "Pos")
print(len(positive_text))
#for r in positive_text:
   # print(r)

negative_text = get_text(reviews, "Neg")
print(len(negative_text))
#for r in negative_text:
   # print(r)

def count_text(text):
  # Split text into words based on whitespace.  Simple but effective.
  words = re.split("\s+", text)
  # Count up the occurence of each word.
  return Counter(words)

negative_counts = count_text(str(negative_text))
positive_counts = count_text(str(positive_text))
print(positive_counts)
print(negative_counts)

print("Positive text sample: {0}".format(positive_text[:100]))
print("Negative text sample: {0}".format(negative_text[:100]))

"""
negative_text = get_text(reviews, "Neg")
print(negative_text)
positive_text = get_text(reviews, "Pos")
# Generate word counts for negative tone.
negative_counts = count_text(negative_text)
# Generate word counts for positive tone.
positive_counts = count_text(positive_text)

print("Negative text sample: {0}".format(negative_text[:100]))
#print("Positive text sample: {0}".format(positive_text[:100]))

"""