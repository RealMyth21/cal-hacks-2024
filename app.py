from flask import Flask, render_template, request, redirect, url_for, jsonify

import boto3
#import pymysql
#import pymysql.cursors
import os
import json
from botocore.exceptions import ClientError
import os
import torch
import torchaudio
import numpy as np
from scipy.io import wavfile
#from BerkeleyAIHackathon import predict_audio, format_result, model, transform


app = Flask(__name__)
test_audio_path = '/Users/charlottelaw/outputGun.wav'
#predicted_label = predict_audio(model, test_audio_path, transform)
app.config['UPLOAD_FOLDER'] = 'static\css/audios'
UPLOAD_FOLDER = 'static\css/audios'
app.config['DEBUG'] = True
model_id = 'anthropic.claude'

# AWS S3 configuration
S3_BUCKET = 'awsbucketai'
s3 = boto3.client('s3', region_name='us-east-2a')  # Replace with your AWS region

# AWS Bedrock configuration
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-2a')  # Replace with your AWS region

# Database configuration
DB_HOST = 'database-1.c3m8asiy4pvd.us-east-2.rds.amazonaws.com'
DB_USER = 'admin'
DB_PASSWORD = 'berkeley'
DB_NAME = 'database-1'
DB_PORT = 3306
def preprocess_text(event, context):
    text = event['text']
    # Implement text preprocessing as needed
    return {'preprocessed_text': text}

def invoke_summarization_model(event, context):
    client = boto3.client('bedrock-runtime', region_name='us-east-2a')
    workflow_id = 'your-bedrock-workflow-id'
    input_data = {
        'text': event['preprocessed_text'],
        'model_id': 'anthropic.claude'  # Replace with the actual model ID if available
    }
    
    # Invoke Bedrock workflow
    response = client.invoke_workflow(
        WorkflowId=workflow_id,
        Input=json.dumps(input_data)
    )
    
    # Process response as needed
    summary = response['summary']
    return {'summary': summary}
'''
def connect_to_rds():
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT,
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except pymysql.MySQLError as e:
        print(f"Error connecting to RDS: {e}")
        return None
'''
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'mp3', 'wav', 'webm'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_file_to_s3(file, bucket_name, acl='public-read'):
    try:
        s3.upload_fileobj(file, bucket_name, file.filename, ExtraArgs={'ACL': acl})
        return True
    except ClientError as e:
        print(f"Error uploading file to S3: {e}")
        return False

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/demo', methods=['GET', 'POST'])
def demo():
    summary = None
    try:
        if request.method == 'POST':
            if 'file' in request.files:
                file = request.files['file']
                if file.filename == '':
                    print("No selected file")
                    return redirect(request.url)
                if file and allowed_file(file.filename):
                    filename = file.filename
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    print(f"File saved: {filename}")


                     # Save the uploaded file to the uploads folder
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                    file.save(file_path)
                    # Perform sentiment analysis on the uploaded audio file
                    predicted_scores = predict_audio(model, file_path, transform)
                    if predicted_scores:
                        # Format the sentiment analysis result
                        formatted_result = format_result(predicted_scores)
                        return render_template('index.html', result=formatted_result)
            

                    #return redirect(url_for('demo'))
            elif 'text' in request.form:
                text = request.form['text']
                if text:
                    # Invoke the Titan model for text summarization
                    response = bedrock_client.invoke_model(
                        modelId='amazon.titan-text-express-v1',  # Use the correct Titan model ID
                        contentType='application/json',
                        body=json.dumps({'text': text})
                       
                        
                    )
                    response_body = json.loads(response['body'].read().decode())
                    summary = response_body['summary']
                    print(f"Summary: {summary}")
        
        return render_template('demo.html', summary=summary)
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred during the process."
'''
def demo():
    summary = None
    try:
        if request.method == 'POST':
            if 'file' in request.files:
                file = request.files['file']
                if file.filename == '':
                    print("No selected file")
                    return redirect(request.url)
                if file:
                    # Upload file to S3
                    if upload_file_to_s3(file, S3_BUCKET):
                        filename = file.filename
                        url = f"https://{S3_BUCKET}.s3.amazonaws.com/{filename}"
                        # Save file details to database or perform other operations
                        print(f"File uploaded to S3 and metadata saved: {filename}")
                        return redirect(url_for('demo'))
                    else:
                        return "Error uploading file to S3"
            elif 'text' in request.form:
                text = request.form['text']
                if not text:
                    return "No text provided", 400
                try:
                    response = bedrock_client.invoke_endpoint(
                        ModelId=model_id,
                        ContentType='application/json',
                        Body=json.dumps({'text': text})
                    )
                    summary = response['Output']
                    print(f"Text summarized: {summary}")
                except Exception as e:
                    print(f"Error: {e}")
                    return "An error occurred during the text summarization process.", 500
        
        return render_template('demo.html', summary=summary)
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred during the process."
    
'''
'''
def demo():
    summary = None
    try:
        if request.method == 'POST':
            if 'file' in request.files:
                file = request.files['file']
                if file.filename == '':
                    print("No selected file")
                    return redirect(request.url)
                if file and allowed_file(file.filename):
                    if upload_file_to_s3(file, S3_BUCKET):
                        filename = file.filename
                        url = f"https://{S3_BUCKET}.s3.amazonaws.com/{filename}"
                        connection = connect_to_rds()
                        if connection:
                            try:
                                with connection.cursor() as cursor:
                                    sql = "INSERT INTO audio_files (filename, url) VALUES (%s, %s)"
                                    cursor.execute(sql, (filename, url))
                                    connection.commit()
                            except Exception as e:
                                print(f"Error saving to database: {e}")
                                return "An error occurred during the file upload process."
                            finally:
                                connection.close()
                        else:
                            return "Could not connect to the database."
                        print(f"File uploaded to S3 and metadata saved: {filename}")
                        return redirect(url_for('demo'))
                    else:
                        return "Error uploading file to S3"
            elif 'text' in request.form:
                text = request.form['text']
                if not text:
                    return "No text provided", 400
                try:
                    response = bedrock_client.generate_summary(
                        Text=text,
                        MaxTokens=100  # Adjust as per your needs
                    )
                    summary = response['Summary']
                except Exception as e:
                    print(f"Error: {e}")
                    return "An error occurred during the summarization process.", 500
        
        return render_template('demo.html', summary=summary)
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred during the process."
'''
if __name__ == '__main__':
    app.run(debug=True)
