from flask import Flask, render_template, request, redirect, url_for,session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import secrets
from flask import send_file
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
import re
from HTTR  import * 
from CV_FUNCTIONS import Pil2cv_converter, Pdf2Image , computer_vision_scanned_version_file_ide, computer_vision_soft_version_file_ide
from flask import send_from_directory
import pandas as pd
from src.utils_logic_generation import checkingg
from utils_asag_inference import *
import os


flask_app = Flask(__name__)
flask_app.secret_key = secrets.token_hex(16)  # Set your secret key for session management


# ps, Httr_model, device, processor = initialize_model(
#     SiameseModel_path="siamese_model.pth",
#     main_model_path="TrOCR-finetuned-iamcvl/TrOCR-finetuned.pt",
# )
Httr_model =  VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-large-handwritten", 
        #device_map = "cpu"
)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
# Httr_model = pickle.load(open("model.pkl", "rb"))
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")

me, sent_model, bert_model, bert_tokenizer, paraphrase_model, wrong_threshold, keyword_model, keyword_tokenizer, random_forest_ensemble = initilize_models_asag()

#inference_asag(['i love cats','i hate you'], ['i like cats','i hate you'], sent_model, bert_model, bert_tokenizer, random_forest_ensemble, me, keyword_model, keyword_tokenizer, stog = None, gtos = None, paraphrase = False) 


# Database configuration using PyMySQL
flask_app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root:11122002@127.0.0.1/autograder"
flask_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
flask_app.config['SESSION_TYPE'] = 'filesystem'

# Initialize the database
db = SQLAlchemy(flask_app)

# Define the Professors model
class Professor(db.Model):
    __tablename__ = 'professors'
    professor_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    birthdate = db.Column(db.Date, nullable=True)
    phone_number = db.Column(db.String(20), nullable=True)
    education_degree = db.Column(db.String(100), nullable=True)

# Define the Student model
class Student(db.Model):
    __tablename__ = 'students'
    student_id = db.Column(db.BigInteger, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    birthdate = db.Column(db.Date, nullable=True)
    phone_number = db.Column(db.String(20), nullable=True)
    
# Create the database tables
with flask_app.app_context():
    db.create_all()



# Home route to render the registration form
@flask_app.route('/')
def home():
    return render_template('login.html')


@flask_app.route('/try')
def overview_page():
    return render_template('overview.html')


@flask_app.route('/home')
def main_page():
    return render_template('index.html')

@flask_app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']

    # Query the database to find the user by email
    user = Professor.query.filter_by(email=email).first()

    if user:
        # Check if the provided password matches the stored password hash
        if check_password_hash(user.password, password):
            # If the credentials are valid, store user information in the session
            session['user_id'] = user.professor_id
            session['username'] = user.username
            # Redirect to a protected route or dashboard
            if  user.email=='admin@gmail.com':
                return render_template('regist.html')
            else:
                return main_page()
    
    # If login fails, redirect back to the login page with a flash message
    flash('Invalid email or password', 'error')
    return home()


@flask_app.route('/register', methods=['POST'])
def register():
    # Retrieve form data
    username = request.form['username']
    email = request.form['email']
    password = generate_password_hash(request.form['password'])  # Hash the password
    name = request.form['name']
    birthdate = request.form['birthdate']
    phone_number = request.form['phone_number']
    education_degree = request.form['education_degree']

    # Create a new professor instance
    new_professor = Professor(
        username=username,
        email=email,
        password=password,
        name=name,
        birthdate=birthdate,
        phone_number=phone_number,
        education_degree=education_degree
    )

    # Add and commit the new professor to the database
    db.session.add(new_professor)
    db.session.commit()

    return redirect(url_for('home'))

@flask_app.route("/predict", methods = ["POST"])
def predict():
    
    torch.cuda.empty_cache()


    # Access text from the textarea
    model_answer_text = request.form['modelAnswerText']

    # Access uploaded file
    
    uploaded_file = request.files.get("studentanswerpdf")
    if uploaded_file.filename == "":
        file_path = "scanned.pdf"

        with open(file_path, "rb") as f:
            file_content = f.read()
            
        # Create a new FileStorage object from the read file content
        from werkzeug.datastructures import FileStorage
        from io import BytesIO
        uploaded_file = FileStorage(
            stream=BytesIO(file_content),
            filename="scanned.pdf",
            content_type="application/pdf"
        )
    # Access uploaded file
    uploaded_file2 = request.files.get("modelanswerpdf")



    if uploaded_file is None:
        return "No studentanswerpdf uploaded!", 400  # Handle missing file error

    if uploaded_file2 is None:
        return "No modelanswerpdf uploaded!", 400  # Handle missing file error


    # Get temporary file path
    studentanswerpdf_path = uploaded_file.filename  # Use the filename for now
    modelanswerpdf_path = uploaded_file2.filename  # Use the filename for now

 # Process the first uploaded pdf 
    studentanswerpdf_pages = Pdf2Image(inputtype=0, path=studentanswerpdf_path)
    studentanswerpdf_pages = [Pil2cv_converter(page) for page in studentanswerpdf_pages]
    num_studentanswerpdf_pages= len(studentanswerpdf_pages)
    studentanswerpdf_page_parts = computer_vision_soft_version_file_ide(studentanswerpdf_pages)
    studentanswers_scanned = computer_vision_scanned_version_file_ide(studentanswerpdf_pages,studentanswerpdf_page_parts)
    (studentanswer_id, studentanswer_name, studentanswer1, studentanswer2, studentanswer3, studentanswer4, studentanswer5, studentanswer6, studentanswer7, studentanswer8, studentanswer9) = studentanswers_scanned
    answers = [studentanswer_id,studentanswer_name,studentanswer1,studentanswer2,studentanswer3,studentanswer4,studentanswer5,studentanswer6,studentanswer7,studentanswer8,studentanswer9]
    student_answers_ocr = [httr_prediction(Httr_model, processor, answer) for answer in answers]


    # Process the second uploaded pdf 
    modelanswerpdf_pages = Pdf2Image(inputtype=0, path=modelanswerpdf_path)
    modelanswerpdf_pages = [Pil2cv_converter(page) for page in modelanswerpdf_pages]
    num_modelanswerpdf_pages= len(modelanswerpdf_pages)
    modelanswerpdf_page_parts = computer_vision_soft_version_file_ide(modelanswerpdf_pages)
    modelanswers_scanned = computer_vision_scanned_version_file_ide(modelanswerpdf_pages,modelanswerpdf_page_parts)
    (modelanswer_id, modelanswer_name, modelanswer1, modelanswer2, modelanswer3, modelanswer4, modelanswer5, modelanswer6, modelanswer7, modelanswer8, modelanswer9) = modelanswers_scanned
    answers2 = [modelanswer_id,modelanswer_name,modelanswer1,modelanswer2,modelanswer3,modelanswer4,modelanswer5,modelanswer6,modelanswer7,modelanswer8,modelanswer9]
    model_answers_ocr = [httr_prediction(Httr_model, processor, answer) for answer in answers2]



    remove_zeros(student_answers_ocr)
    remove_zeros(model_answers_ocr)
   


    if model_answer_text:
        grades =inference_asag(student_answers_ocr,model_answers_ocr, sent_model, bert_model, bert_tokenizer, random_forest_ensemble, me, keyword_model, keyword_tokenizer, stog = None, gtos = None, paraphrase = False) 
        print("model answer grades are ready")
    else:
        grades = inference_asag(student_answers_ocr,model_answers_ocr, sent_model, bert_model, bert_tokenizer, random_forest_ensemble, me, keyword_model, keyword_tokenizer, stog = None, gtos = None, paraphrase = False) 
        print("model answer grades are ready")

    _, _, q1Grade, q2Grade, q3Grade, q4Grade, q5Grade, q6Grade, q7Grade, q8Grade, q9Grade = grades




    (individual_model_id, individual_model_name, individual_model_answer1, individual_model_answer2, individual_model_answer3, individual_model_answer4, individual_model_answer5, individual_model_answer6, individual_model_answer7, individual_model_answer8, individual_model_answer9) = model_answers_ocr

         
     


    
  
    global individual_student_id
    (individual_student_id, individual_student_name, individual_student_answer1, individual_student_answer2, individual_student_answer3, individual_student_answer4, individual_student_answer5, individual_student_answer6, individual_student_answer7, individual_student_answer8, individual_student_answer9) = student_answers_ocr
    
    if individual_student_id.startswith('0') and individual_student_id.endswith('0') :
       # Query the student by name
        student_by_name = Student.query.filter_by(name=individual_student_name).first()
        if student_by_name:
            individual_student_id= student_by_name.student_id
        else:
            print(student_by_name)
       
    elif individual_student_name.startswith('0') and individual_student_name.endswith('0') :
        student_by_id = Student.query.filter_by(student_id=individual_student_id).first()
        if student_by_id :
            individual_student_name=student_by_id.name
            #individual_student_id=checkingg( individual_student_id,get_student_ids())
    #else:
        #individual_student_id=checkingg( individual_student_id,get_student_ids())

       

    bothanswers_result = bothanswers_function(
        individual_student_name,individual_student_id,num_studentanswerpdf_pages,
        individual_student_answer1, individual_model_answer1,q1Grade,
        individual_student_answer2, individual_model_answer2,q2Grade,
        individual_student_answer3, individual_model_answer3,q3Grade,
        individual_student_answer4, individual_model_answer4,q4Grade,
        individual_student_answer5, individual_model_answer5,q5Grade,
        individual_student_answer6, individual_model_answer6,q6Grade,
        individual_student_answer7, individual_model_answer7,q7Grade,
        individual_student_answer8, individual_model_answer8,q8Grade,
        individual_student_answer9, individual_model_answer9,q9Grade,
    )

    df = pd.DataFrame({'student_answers': student_answers_ocr, 'model_answers': model_answers_ocr, 'grades': grades})
    df_trimmed = df.iloc[2:]
    file_name = f'{individual_student_id}.csv'
    df_trimmed.to_csv(file_name, index=False)

  

    return bothanswers_result






def bothanswers_function(
        
        individual_student_name,individual_student_id,num_studentanswerpdf_pages,

        individual_student_answer1, individual_model_answer1,q1Grade,
        individual_student_answer2, individual_model_answer2,q2Grade,
        individual_student_answer3, individual_model_answer3,q3Grade,
        individual_student_answer4, individual_model_answer4,q4Grade,
        individual_student_answer5, individual_model_answer5,q5Grade,
        individual_student_answer6, individual_model_answer6,q6Grade,
        individual_student_answer7, individual_model_answer7,q7Grade,
        individual_student_answer8, individual_model_answer8,q8Grade,
        individual_student_answer9, individual_model_answer9,q9Grade,
    ):
    
    return render_template(
        "New Text Document.html",
        numquestions=num_studentanswerpdf_pages * 3,dispstudentName = individual_student_name , dispstudentID = individual_student_id,
        
        dispstudentans1=individual_student_answer1, dispmodelans1=individual_model_answer1, dispgrade1=q1Grade,
        dispstudentans2=individual_student_answer2, dispmodelans2=individual_model_answer2, dispgrade2=q2Grade,
        dispstudentans3=individual_student_answer3, dispmodelans3=individual_model_answer3, dispgrade3=q3Grade,
        dispstudentans4=individual_student_answer4, dispmodelans4=individual_model_answer4, dispgrade4=q4Grade,
        dispstudentans5=individual_student_answer5, dispmodelans5=individual_model_answer5, dispgrade5=q5Grade,
        dispstudentans6=individual_student_answer6, dispmodelans6=individual_model_answer6, dispgrade6=q6Grade,
        dispstudentans7=individual_student_answer7, dispmodelans7=individual_model_answer7, dispgrade7=q7Grade,
        dispstudentans8=individual_student_answer8, dispmodelans8=individual_model_answer8, dispgrade8=q8Grade,
        dispstudentans9=individual_student_answer9, dispmodelans9=individual_model_answer9, dispgrade9=q9Grade,
    )





from flask import Flask, send_file


#### will change (if student_id and name ) to be comparred by 00000 depend on cv output
# Function to check student details

#get ids from db
def get_student_ids():
    student_ids = [student.student_id for student in Student.query.all()]
    return student_ids
#checkingg(id,get_student_ids())

def remove_zeros(lst):
    for i in range(2, len(lst)):
        if "0" in lst[i]:
            lst[i] = lst[i].replace("0", "")


@flask_app.route('/download')
def download_file():
    file_path = f'{individual_student_id}.csv'
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    flask_app.run(debug=True, use_reloader =False)
