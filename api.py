from flask import Flask,request,jsonify
from flask_cors import CORS
from gradio_predict import correct


app = Flask(__name__)
CORS(app)

@app.route("/predict",methods=['POST'])
def predict():
    text = request.json["text"]
    try:
        out = correct(text)
        return jsonify({"result":out})
    except Exception as e:
        print(e)
        return jsonify({"result":"Model Failed"})

if __name__ == "__main__":
    app.run('0.0.0.0',port=8000)