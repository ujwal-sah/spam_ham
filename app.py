from flask import Flask,render_template,request
import pickle

# load the model from disk

clf = pickle.load(open('nb.pkl', 'rb'))
cv = pickle.load(open('countvect.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    message = request.form['message']
    data = [message]
    vect = cv.transform(data)
    my_prediction = clf.predict(vect)
    print(my_prediction)
    
    if my_prediction == 1:
        prediction="a spam"
    elif my_prediction == 0:
        prediction="not a spam"
    return render_template('index.html',prediction_text = "It is " + prediction +" message")

if __name__ == '__main__':
	app.run()