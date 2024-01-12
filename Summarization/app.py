from flask import *
#from flask_restful import Api, Resources
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import jsonpickle
import requests
from bs4 import BeautifulSoup
import csv
import pickle 
import time
from datetime import datetime
import sentencepiece as spm
import tensorflow.compat.v1 as tf
import tensorflow as tf
import tensorflow_hub as hub
import json
import cgi
from transformers import pipeline

app = Flask(__name__)

# sumarization pipeline loads
# Available Models list
available_summarize_model = ["facebook/bart-large-cnn"]
s_model_selected = available_summarize_model[0]
pipe = pipeline("summarization", model=s_model_selected)

streaming = True
max_output_tokens = 200

# Available Models list
available_model_apis = [$LLM_MODEL]
# Let's select the model from available list
model_selected = available_model_apis[0]

def sumarizedmodel_api(issues):
    print("Inside sumarizedmodel_api")
    result = pipe(issues, max_length=147, min_length=100, do_sample=True)
    summary = result[0]['summary_text'].replace(u'\xa0', u'')
    return summary


def stream_and_yield_response(response):
    for chunk in response.iter_lines():
        decoded_chunk = chunk.decode("utf-8")
        if decoded_chunk == "data: [DONE]":
            pass
        elif decoded_chunk.startswith("data: {"):
            payload = decoded_chunk.lstrip("data:")
            json_payload = json.loads(payload)
            yield json_payload['choices'][0]['text']


def llm_api(data):
    """
    Creates a request to LLM model with API key in header.
    """
    # custom
    output = ""
    #
    url = $URL
      
    headers = {
        'accept': 'application/json',
        'api-key': $API_KEY,
        'Content-Type': 'application/json'
    }
      
    try:
        response = requests.post(url, headers=headers, json=data, stream=data['stream'], verify=False)
        response.raise_for_status()

        if data['stream']:
            print("Inside IF")
            for result in stream_and_yield_response(response):
                print(result, end='')
                # custom
                output = output + result
                #                
        else:
            print("Inside ELSE")
            response_dict = response.json()
            result = response_dict['choices'][0]['text']
            #print(result)
            output = output + result
        return output
        
    except requests.exceptions.HTTPError as err:
        print('Error code:', err.response.status_code)
        print('Error message:', err.response.text)
        output = err.response.text
    except Exception as err:
        print('Error:', err)
        output = err

# API Call
# json_payload = llm_api(data) 

def issue_processing(issues):
    """
    Creates a request to LLM model with API key in header.
    """
    print("%%%%%%%%%%%Inside LLM issue_processing")
    output = ""
    
    url = $URL
      
    headers = {
        'accept': 'application/json',
        'api-key': $API_KEY,
        'Content-Type': 'application/json'
    }

        # Model instruction and Parameters
        # persona --> context --> task --> tone --> format
        # I am a security manager and your task is to generate a short summary of 
        # following paragraphs of security vulnerabilities. Summarize each in at most 30 words
        # Your task is to generate a short summary of a product \
        # review of an application that is available on playstore.
 

    #instruction_text = f'Can you summarize {issues} in less than 500 tokens?'
    #instruction_text = f'Can you brief short summarize the below paragraph \n {issues} ?'
    instruction_text = f'I am a security manager and your task is to generate a short summary of following paragraphs of security vulnerabilities. Summarize each paragraph in at most 30 words \n {issues} ?' 
          
    data = {
        'prompt': f'{instruction_text}',
        'temperature': 0.5,
        'top_p': 0.95,
        'max_tokens': f'{max_output_tokens}',
        'stream': streaming,
        'model': f'{model_selected}'
        }

      
    try:
        response = requests.post(url, headers=headers, json=data, stream=data['stream'], verify=False)
        response.raise_for_status()

        if data['stream']:
            print("Inside IF")
            for result in stream_and_yield_response(response):
                print(result, end='')
                output = output + result
            
            print("result ===>",result)        
        else:
            print("Inside ELSE")
            response_dict = response.json()
            result = response_dict['choices'][0]['text']
            print(result)
        
        output =  output.replace('\n','<br>')
        output =  output.replace('\"','')
        #output = unicodeToHTMLEntities(output)
        
        return output
        
    except requests.exceptions.HTTPError as err:
        print('Error code:', err.response.status_code)
        print('Error message:', err.response.text)
        result = err.response.text
    except Exception as err:
        print('Error:', err)
        result = err
        
        
def unicodeToHTMLEntities(text):
    """Converts unicode to HTML entities.  For example '&' becomes '&amp;'."""
    text = cgi.escape(text).encode('ascii', 'xmlcharrefreplace')
    return text

def issue_processing1(issues):
    try:    
        print("Inside LLM issue_processing1...")
        # process code might be direct Summarization API or use LLM APIs?
        
        # Model instruction and Parameters
        instruction_text = f'Please provide a short and concise summary of the following security issues \n {issues} ?'
          
        data = {
            'prompt': f'{instruction_text}',
            'temperature': 0.5,
            'top_p': 0.95,
            'max_tokens': f'{max_output_tokens}',
            'stream': streaming,
            'model': f'{model_selected}'
            }

        # API Call
        result = llm_api(data)         
        print("result ===>",result)        
        return result
    except Exception as e:
        print(f"Error Occured while cve_processing- ,{e}")
        return "Error Occured while cve_processing";

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_issues = request.form.get('name')
        llm_flag = request.form.get('i_llm')
        #input_CVE_desc = request.form.get('desc')
        print('llm_flag values ==> ',llm_flag)
        print('Input Issues are ============> ',input_issues)
        #result = issue_processing1(input_issues)
        result =""
        if llm_flag == "true":
            result = issue_processing1(input_issues)
        else:
            result = sumarizedmodel_api(input_issues)
        #&&&&&&&&&&&&
        #result = unicodeToHTMLEntities(result)
        result =  result.replace('\n','<br>')
        result =  result.replace('\"','')
        result =  result.replace('\u2022','-')
        #print("=======> ",result)
        tbl_tag="";
        if "Error Occured during method issue_processing" in result:
            tbl_tag = tbl_tag +"<tr><td>Error Occured, Please try again</td></tr>";
        else:
            print("No Error so here is the result :) ", result)
            tbl_tag = tbl_tag +"<tr><td>"+result+"</td></tr>"
            #print("inside == ",tbl_tag)        
        

        
        
        print("tbl_tag ==> ",tbl_tag)
        tbl_tag = "<tr><th bgcolor='#009879'>Summary &nbsp;&nbsp;</th></tr>"+tbl_tag
        return jsonpickle.encode(tbl_tag)



if __name__ == '__main__':
    app.run(debug=True)
