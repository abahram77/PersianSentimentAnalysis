from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer_digi = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-digikala")

model_digi = AutoModelForSequenceClassification.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-digikala")

tokenizer_snappfood = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-snappfood")

model_snappfood = AutoModelForSequenceClassification.from_pretrained(
    "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood")

import numpy
import torch


def clean(text):
    import re
    import string
    tokenized_text = ""
    for letter in text:
        m = re.search('^[آ-ی]$', letter)
        if (m is None and letter not in ["؟", ".", "!", "?", "!", "."]):
            tokenized_text += " "
        else:
            tokenized_text += letter

    return (" ".join(tokenized_text.split()))


def sentiment_snappfood(text):
    text = clean(text)
    classes = ["positive", "negative"]
    tokenized_text = tokenizer_snappfood.encode_plus(text, return_tensors="pt")
    tokenized_text_classification_logits = model_snappfood(**tokenized_text)[0]
    results = torch.softmax(tokenized_text_classification_logits, dim=1).tolist()[0]

    # if(results[0]>results[1]):
    #   return(results,f"Positive  with {max(results)} probability")
    # else:
    #   return(results,f"Negative with {max(results)} probability")
    if (results[0] > results[1]):
        return (1)
    else:
        return (0)

from  Dataset import x,y

def sentiment_digi(text):
    text = clean(text)
    classes = ["negative", "neutrual", "positive"]
    tokenized_text = tokenizer_digi.encode_plus(text, return_tensors="pt")
    tokenized_text_classification_logits = model_digi(**tokenized_text)[0]
    results = torch.softmax(tokenized_text_classification_logits, dim=1).tolist()[0]

    # if(results[0]>results[1] and results[0]>results[2]):
    #   return("Neutral",max(results))
    # elif(results[1]>results[0] and results[1]>results[2]):
    #   return("Negative",max(results))
    # elif(results[2]>results[0] and results[2]>results[1]):
    #   return("Positive",max(results))

    if (results[1] > results[2]):
        return (0)
    elif (results[2] > results[1]):
        return (1)


def accuracy_snp_food(x, y):
    correct = 0
    for i in range(len(x)):
        if (sentiment_snappfood(x[i]) == y[i]):
            correct += 1
    print("accuracy snappfood Data: ", correct / len(x))


def accuracy_digi(x, y):
    correct = 0
    for i in range(len(x)):
        if (sentiment_digi(x[i]) == y[i]):
            correct += 1
    print("accuracy digikala Data: ", correct / len(x))



accuracy_snp_food(x, y)

accuracy_digi(x, y)

