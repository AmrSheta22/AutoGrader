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
from HTTR import httr_prediction
#from PdfToImage import Pdf2Image
#from PilToCv import Pil2cv_converter
from CV_FUNCTIONS import Pil2cv_converter, Pdf2Image , computer_vision_scanned_version_file_ide, computer_vision_soft_version_file_ide
#from ComputerVision import computer_vision_soft_version ,computer_vision_scanned_version
from Asag import pridector
from flask import send_from_directory
import pandas as pd
from src.utils_logic_generation import checkingg
import os

flask_app = Flask(__name__)
flask_app.secret_key = secrets.token_hex(16)  # Set your secret key for session management


Httr_model = pickle.load(open("model.pkl", "rb"))
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
Asag_model = SentenceTransformer("Zingy_modeling", device='cpu')



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
    return main_page()


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

    #dimensions
    imgs_dim = Pdf2Image(inputtype=0, path="Form_for_dimensions.pdf")     
    imgs_dim=[Pil2cv_converter(img_dim) for img_dim in imgs_dim]
    answerss, countorss, heightss, widthss = computer_vision_soft_version_file_ide(imgs_dim) 


    # Access text from the textarea
    model_answer_text = request.form['modelAnswerText']

    # Access uploaded file
    uploaded_file = request.files.get("studentanswerpdf")

    # Access uploaded file
    uploaded_file2 = request.files.get("modelanswerpdf")


    if uploaded_file is None:
        return "No studentanswerpdf uploaded!", 400  # Handle missing file error

    if uploaded_file2 is None:
        return "No modelanswerpdf uploaded!", 400  # Handle missing file error


    # Get temporary file path
    pdf_path = uploaded_file.filename  # Use the filename for now
    pdf_path2 = uploaded_file2.filename  # Use the filename for now



    # Process the PDF using your Pdf2Image function
    images = Pdf2Image(inputtype=0, path=pdf_path)
    images2 = Pdf2Image(inputtype=0, path=pdf_path2)
    numpages_stuedntans = len(images)
    numpages_modelans =len(images2)
    image_cv=[Pil2cv_converter(image) for image in images]
    image_cv2=[Pil2cv_converter(image) for image in images2]

            
    
    answers= [computer_vision_scanned_version_file_ide(img, countorss, heightss, widthss) for img in image_cv]
    answers2= [computer_vision_scanned_version_file_ide(img, countorss, heightss, widthss) for img in image_cv2]



    student_answers = [httr_prediction(Httr_model, processor, answer) for answer in answers]
    ## check id correct , check data intg
    model_answers = [httr_prediction(Httr_model, processor, answer) for answer in answers2]


    if model_answer_text:
        grades = [pridector(Asag_model, student_answer, model_answer_text, 'easy') for student_answer in student_answers]
    else:
        grades = [pridector(Asag_model, student_answer, model_answer, 'easy') for student_answer, model_answer in zip(student_answers, model_answers)]


 

    if model_answer_text:
        individual_model_answer = model_answer_text
    else:
        for model_ans in model_answers:
            individual_model_answer = model_ans
    
    



    for student_ans in student_answers:
        individual_student_answer = student_ans

    for grade in grades:
        individual_grade = grade
 

  
  
    bothanswers_result = bothanswers_function(individual_student_answer,individual_model_answer,individual_grade,numpages_stuedntans)

    df = pd.DataFrame({'student_answers': student_answers, 'model_answers': model_answers, 'grades': grades})
    df.to_csv('student1.csv', index=False) 

  

    return bothanswers_result



def bothanswers_function(individual_student_answer ,individual_model_answer ,individual_grade,numpages_stuedntans):
    return render_template("New Text Document.html", dispstudentans = individual_student_answer ,dispmodelans = individual_model_answer ,dispgrade = individual_grade ,numquestions = numpages_stuedntans)



from flask import Flask, send_file


#### will change (if student_id and name ) to be comparred by 00000 depend on cv output
# Function to check student details
def check_student_details(student_id=None, name=None):
    if student_id: #!=''
        student = Student.query.filter_by(student_id=student_id).first()
        if student:
            if student.name:
                return student.name  # if id exist return name
        
    elif name: #!=''
        student_by_name = Student.query.filter_by(name=name).first()
        if student.student_id:
            return student.student_id

#get ids from db
def get_student_ids():
    student_ids = [student.student_id for student in Student.query.all()]
    return student_ids
#checkingg(id,get_student_ids())

@flask_app.route('/download')
def download_file():
    file_path = 'student1.csv'
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    flask_app.run(debug=True)
