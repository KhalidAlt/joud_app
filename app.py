from flask import Flask, render_template, redirect, url_for
from flask import request, jsonify
from datasets import load_dataset
from flask_apscheduler import APScheduler
from collections import Counter
import random
import json
import os
import subprocess


# set configuration values
class Config:
    SCHEDULER_API_ENABLED = True

app = Flask(__name__)
app.config.from_object(Config())

scheduler = APScheduler()

class Config:
    SCHEDULER_API_ENABLED = True

scheduler.init_app(app)

def get_finished_indices():
    if os.path.exists('static/data/dataset.json'):
        with open('static/data/dataset.json') as f:
            data = json.load(f)

        finished_indices = []
        for element in data:
            finished_indices.append(int(element['index']))
        return set(finished_indices)
    else:
        return set()

def load_data():
    ds = load_dataset('asas-ai/joud_sample','all')
    ds = ds.map(lambda example: {"text": example["text"], "dataset_name": example["meta"]["dataset_name"]}, remove_columns=['meta'])
    index = list(range(0,len(ds['train'])))
    ds['train'] = ds['train'].add_column("index", index)
    # all instructions
    all_indices = set([i for i in range(len(ds['train']))])

    return all_indices, ds

def save_json(entry):
    data = []
    if os.path.exists('static/data/dataset.json'):
        with open('static/data/dataset.json') as f:
            data = json.load(f)

    data.append(entry)

    with open('static/data/dataset.json', 'w') as f:
        json.dump(data, f, ensure_ascii = False, indent=2)

all_indices, ds = load_data()


@app.route('/api/submit',methods = ['POST', 'GET'])
def submit():
    if request.method == 'POST':
        element = {k:request.form[k] for k in request.form}
        save_json(element)
    return render_template('index.html')

@app.route('/api/data')
def send_data():
    finished_indices = get_finished_indices()
    rem_indices = all_indices - finished_indices
    index = random.choice(list(rem_indices))
    element = ds['train'][index]
    # for key in prob_mt_ar:
    #     element['instruction'] = element['instruction'].replace(key, prob_mt_ar[key], 1)
    element['num_rem'] = len(rem_indices)
    return jsonify(element)

@app.route('/api/getConNames')
def get_cont_names():
    with open('static/data/dataset.json') as f:
        data = json.load(f)
    return jsonify(Counter([elm['Reviewed by'].strip().split(' ')[0].strip() for elm in data]))

@app.route('/api/getCon', methods = ['POST', 'GET'])
def get_cont():
    print(request.form)
    name = request.form['Reviewed by']
    with open('static/data/dataset.json') as f:
        data = json.load(f)
    return jsonify({
        "num_cont":len([elm for elm in data if elm['Reviewed by'] == name])
    })


@app.route('/api/saved')
def send_saved_data():
    element = {
            "output" :'',
            "num_rem":0,
            "index":-1
        }
    with open('static/data/dataset.json') as f:
        data = json.load(f)
    if len(data):
        saved_indices = list(range(len(data)))
        index = random.choice(saved_indices)
        element = data[index]
        element['num_rem'] = len(saved_indices)
    return jsonify(element)

@scheduler.task('interval', id='do_push_hf', hours=1)
def push_hub():
    TOKEN = os.environ.get('HF_TOKEN')
    subprocess.run(["huggingface-cli", "login", "--token", TOKEN])
    with open('static/data/dataset.json') as f:
        data = json.load(f)

    if len(data):
        dataset = load_dataset("json", data_files="static/data/dataset.json",  download_mode = "force_redownload")
        dataset.push_to_hub('asas-ai/joud_cleaned_sample')

def init_dataset():
    os.makedirs('static/data', exist_ok=True)
    try:
        print('loading previous dataset')
        ds = load_dataset('asas-ai/joud_cleaned_sample', download_mode = "force_redownload", verification_mode='no_checks')
        data = [elm for elm in ds['train']]
    except:
        data = []

    with open('static/data/dataset.json', 'w') as f:
        json.dump(data, f, ensure_ascii = False, indent=2)

@app.route('/explore')
def explore():
    return render_template('explore.html')


@app.route('/')
def index():
    return render_template('index.html')

init_dataset()
scheduler.start()

if __name__ == '__main__':
    app.run(port=5000)
