-- Create Professors table
CREATE TABLE Professors (
    professor_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    birthdate DATE,
    phone_number VARCHAR(20),
    education_degree VARCHAR(100)
);

-- Create Students table
CREATE TABLE Students (
    student_id BIGINT  PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    birthdate DATE,
    phone_number VARCHAR(20)
);

-- Create Courses table
CREATE TABLE Courses (
    course_id INT AUTO_INCREMENT PRIMARY KEY,
    course_name VARCHAR(100) NOT NULL,
    professor_id INT,
    FOREIGN KEY (professor_id) REFERENCES Professors(professor_id)
);

-- Create Enrollments table (Many-to-Many relationship between Students and Courses)
CREATE TABLE Enrollments (
    enrollment_id INT AUTO_INCREMENT PRIMARY KEY,
    student_id BIGINT,
    course_id INT,
    FOREIGN KEY (student_id) REFERENCES Students(student_id),
    FOREIGN KEY (course_id) REFERENCES Courses(course_id)
);

-- Create Quizzes table
CREATE TABLE Quizzes (
    quiz_id INT AUTO_INCREMENT PRIMARY KEY,
    course_id INT,
    quiz_name VARCHAR(100),
    quiz_date DATE,
    pdf_path VARCHAR(255),
    FOREIGN KEY (course_id) REFERENCES Courses(course_id)
);

-- Create QuizResults table
CREATE TABLE QuizResults (
    result_id INT AUTO_INCREMENT PRIMARY KEY,
    quiz_id INT,
    student_id BIGINT,
    score DECIMAL(5, 2),
    FOREIGN KEY (quiz_id) REFERENCES Quizzes(quiz_id),
    FOREIGN KEY (student_id) REFERENCES Students(student_id)
);
