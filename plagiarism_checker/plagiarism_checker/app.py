from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

def detect(input_text):
    vectorized_text = tfidf_vectorizer.transform([input_text])
    result = model.predict(vectorized_text)
    return "Plagiarism Detected" if result[0] == 1 else "No Plagiarism"

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    try:
        # Check if 'input_text' is present in form data
        input_text = request.form.get('input_text')
        if not input_text:
            return render_template('index.html', error="No input text provided.")

        # Detect plagiarism and render the result
        result = detect(input_text)
        return render_template('index.html', input_text=input_text, result=result)

    except KeyError as e:
        return render_template('index.html', error=f"Missing form key: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
