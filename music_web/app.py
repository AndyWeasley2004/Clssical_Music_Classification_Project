from flask import Flask, jsonify, request, render_template
import util
import os

app = Flask(__name__)


@app.route('/')
def root():
    return render_template('app.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    files = os.listdir('/Users/yaomingyang/PycharmProjects/flaskProject/files/')

    file_count = sum(1 for file in files if os.path.isfile(os.path.join('/files', file)))

    filename = f'files/file{file_count+1}.midi'
    file.save(filename)

    return 'File successfully uploaded return'


@app.route('/result', methods=['GET'])
def classify_file():
    file_count = os.listdir('/Users/yaomingyang/PycharmProjects/flaskProject/files/')
    sorted(file_count, reverse=True)
    name = file_count[0]

    encoded = util.encode_midi('files/'+name)

    response = jsonify({
        'composer_identified': util.classify(encoded)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    util.load_model()
    app.run(debug=True, port=8000)
