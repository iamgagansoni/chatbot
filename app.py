from flask import Flask,jsonify,request
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle   #portable serialized objects
import numpy as np

app=Flask(__name__)
@app.route("/bot",method=["POST"])

def responce():
    query = dict(request.form)['query']
    
    from keras.models import load_model
    model = load_model('chatbot_model.h5')
    import json
    import random
    intents = json.loads(open('datas.json').read())
    words = pickle.load(open('words.pkl','rb'))
    classes = pickle.load(open('classes.pkl','rb'))
    def clean_up_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(sentence, words):
        sentence_words = clean_up_sentence(sentence)
        bag = [0]*len(words) 
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s: 
                    bag[i] = 1
        return(np.array(bag))

    def predict_class(sentence, model):
        p = bow(sentence, words)
        res = model.predict(np.array([p]))[0]   #predicting probablity of tag 
        ERROR_THRESHOLD = 0.25
        results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"datas": classes[r[0]], "probability": str(r[1])})
        return return_list


    def getResponse(ints, intents_json):
        tag = ints[0]['datas']
        list_of_intents = intents_json['datas']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responces'])
                break
        return result

    ints = predict_class(query, model)
    res = getResponse(ints, intents)
    return jsonify({"responce":res})

if __name__=="__main__":
    app.run(host="0.0.0.0",)
