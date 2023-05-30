import urllib.request
from flask import Flask, request, jsonify, abort
from werkzeug.utils import secure_filename
from sklearn.neighbors import NearestNeighbors
import tensorflow_hub as hub
import re
import numpy as np
import logging
import json
import os
import fitz

logging.basicConfig(level=logging.DEBUG)

def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, f'{output_path}')

def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page-1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    page_nums = []
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i+word_length]
            if (i+word_length) > len(words) and (len(chunk) < word_length) and (
                    len(text_toks) != (idx+1)):
                text_toks[idx+1] = chunk + text_toks[idx+1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[Page no. {idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


class SemanticSearch:

    def __init__(self):
        self.use = hub.load(
            'https://tfhub.dev/google/universal-sentence-encoder/4')
        self.fitted = False

    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True

    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]

        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors

    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i+batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings


def load_recommender(path):
    global recommender
    with open(path, 'r') as file:
        content = file.read()
    recommender.fit(eval(content))
    return 'Corpus Loaded.'


def generate_prompt(question):
    topn_chunks = recommender(question)
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'

    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. "\
        "Cite each reference using [number] notation (every result has this number at the beginning). "\
        "Citation should be done at the end of each sentence. If the search results mention multiple subjects "\
        "with the same name, create separate answers for each. Only include information found in the results and "\
        "don't add any additional information. Make sure the answer is correct and don't output false content. "\
        "If the text does not relate to the query, simply state 'Found Nothing'. Ignore outlier "\
        "search results which has nothing to do with the question. Only answer what is asked. The "\
        "answer should be short and concise. \n\nQuery: {question}\nAnswer: "

    prompt += f"Query: {question}\nAnswer:"
    return prompt


recommender = SemanticSearch()


# rest api
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'pdfChunks'


@app.route('/prompt', methods=['GET'])
def get_prompt():
    try:
        question = request.args.get('question')
        filename = request.args.get('filename')
        if not question.strip():
            raise Exception("Question field is empty")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename+'.json')

        if not os.path.exists(file_path):
            return jsonify({"error": 'file does not exist'})

        load_recommender(file_path)

        prompt = generate_prompt(question)

        return jsonify({"prompt": prompt})

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": "An error occurred"})


@app.route('/chunks', methods=['GET'])
def get_chunks():
    try:
        filename = request.args.get('filename')
        uploadPath = app.config['UPLOAD_FOLDER']
        file_path = os.path.join(uploadPath, filename+'.json')

        with open(file_path, 'r') as file:
            content = file.read()

        return jsonify({"chunks": eval(content)})

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": "An error occurred"})


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        url = request.form.get('url')
        file=None
        if not url: 
            file = request.files['file']
        name_to_save = request.args.get('filename')
        if url == None and file == None:
            return jsonify({"error": "Both URL and PDF is empty. Provide atleast one."})

        filename = secure_filename(name_to_save)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if url!=None:
            download_pdf(url, filepath)
        else:
            file.save(filepath)

        text = pdf_to_text(filepath)
        chunks = text_to_chunks(text)

        with open(f'{filepath}.json', 'w') as chunkfile:
            json.dump(chunks, chunkfile)

        # Check if the file exists
        if os.path.exists(filepath):
            # Delete the file
            os.remove(filepath)
            print(f"File '{filepath}' deleted successfully.")
        else:
            print(f"File '{filepath}' does not exist.")

        return jsonify({"chunks": chunks})

    except Exception as e:
        app.logger.error(e)
        return jsonify({"error": "An error occurred"})


@app.route('/delete', methods=['POST'])
def delete_file():
    filename = request.args.get('filename')

    if not filename:
        return jsonify({"error": "Invalid request. Please provide the 'filename' parameter."})

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename+'.json')

    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"message": f"File '{filename}' has been deleted."})
    else:
        return jsonify({"error": f"File '{filename}' does not exist."})

@app.errorhandler(400)
def handle_bad_request(error):
    # Handle the 400 error and return a custom response
    return 'Bad Request: ' + error.description, 400

if __name__ == '__main__':
    app.run(debug=True)
