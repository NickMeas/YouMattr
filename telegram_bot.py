import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from telegram.ext import CommandHandler, MessageHandler, Filters, Updater
import pandas as pd
import numpy as np
import random
import re
import json

model = load_model("./model/best_model.h5")
with open("./data/json/intents_final.json", encoding="utf8") as f:
    dataset = json.load(f)
    
df = pd.DataFrame(dataset["intents"])

dic = {"tag":[], "patterns":[], "responses":[]}
for i in range(len(df)):
    patterns = df[df.index == i]['patterns'].values[0]
    responses = df[df.index == i]['responses'].values[0]
    tags = df[df.index == i]['tag'].values[0]
    for j in range(len(patterns)):
        dic['tag'].append(tags)
        dic['patterns'].append(patterns[j])
        dic['responses'].append(responses)
        
df = pd.DataFrame.from_dict(dic)

tokenizer = Tokenizer(lower=True, split=' ')
tokenizer.fit_on_texts(df['patterns'])
tokenizer.get_config()

patterns2seq = tokenizer.texts_to_sequences(df['patterns'])
x = pad_sequences(patterns2seq, padding='post')

lbl_enc = LabelEncoder()
lbl_enc.fit_transform(df['tag'])

def start(updater, context):
    updater.message.reply_text("Hello there! I am your personal mental health chatbot.")

def help_(updater, context):
    updater.message.reply_text("W.A.L.L.Y. is created to accompany those with severe social anxiety or simply as a robotic buddy\
        for people who are feeling lonely but with the data that W.A.L.L.Y. has access to, please do not fully depend on it to help cure mental illness problems.\
        \n This project is tasked to help those whose voices are not heard simply due to fear of being judged by others \
        \n\n -Nak")    
    
def wally(updater, context):
    text = []
    txt = re.sub('[^a-zA-Z\']', ' ', updater.message.text)
    txt = txt.lower()
    txt = txt.split()
    txt = " ".join(txt)
    text.append(txt)
        
    x_test = tokenizer.texts_to_sequences(text)
    x_test = np.array(x_test).squeeze()
    try:
        x_test = pad_sequences([x_test], padding='post', maxlen=x.shape[1])
    except:
        x_test = pad_sequences([[x_test]], padding='post', maxlen=x.shape[1])
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax()
    tag = lbl_enc.inverse_transform([y_pred])[0]
    responses = df[df['tag'] == tag]['responses'].values[0]

    updater.message.reply_text(random.choice(responses))


def main():
    updater = Updater("7163968971:AAHzuNaDWXlOCrrUna-hDf5arT7T3zlsuS8")
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_))

    dispatcher.add_handler(MessageHandler(Filters.text, wally))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()