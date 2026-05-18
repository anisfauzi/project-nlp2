from flask import Flask, request, render_template


app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html")


@app.route('/word_vector')
def word_vector():
	return render_template("word_vector.html")

if __name__ == '__main__':
    app.run(debug=True)