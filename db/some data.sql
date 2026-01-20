-- Insert more sample data into Professors table
INSERT INTO Professors (username, email, password, name, birthdate, phone_number, education_degree)
VALUES
('prof_emily', 'emily@example.com', 'securepassword789', 'Emily Davis', '1978-09-10', '2233445566', 'PhD in Physics'),
('prof_michael', 'michael@example.com', 'securepassword012', 'Michael Miller', '1982-12-05', '3344556677', 'PhD in Chemistry');

-- Insert more sample data into Students table
INSERT INTO Students (student_id, name, birthdate, phone_number)
VALUES
(20201010791, 'Anna Wilson', '2001-04-17', '5556667777'),
(20201448026, 'James Moore', '2002-02-25', '6667778888'),
(20201088781, 'Lucas Taylor', '2000-09-30', '7778889999'),
(20201110296, 'Olivia Anderson', '1999-11-11', '8889990000');

-- Insert more sample data into Courses table
INSERT INTO Courses (course_name, professor_id)
VALUES
('Physics 101', 1),
('Chemistry 101', 2);

-- Insert more sample data into Enrollments table
INSERT INTO Enrollments (student_id, course_id)
VALUES
(20201010791, 3), (20201448026, 4),
(20201448026, 3), (20201110296, 4),
(20201088781, 3), (20201010791, 4),
(20201110296, 3), (20201088781,4)
;

-- Insert more sample data into Quizzes table
INSERT INTO Quizzes (course_id, quiz_name, quiz_date, pdf_path)
VALUES
(3, 'Final Exam', '2024-06-10', '/path/to/final_exam_compsci101.pdf'),
(3, 'Midterm Exam', '2024-06-05', '/path/to/midterm_exam_math101.pdf'),
(4, 'Quiz 1', '2024-05-20', '/path/to/quiz1_physics101.pdf'),
(4, 'Quiz 2', '2024-06-01', '/path/to/quiz2_chemistry101.pdf');
-- Insert sample data into QuizResults table
INSERT INTO QuizResults (quiz_id, student_id, score)
VALUES
-- Course 1 (Computer Science 101) with Quiz 1 and 2
(1, 20201010791, 88.00), -- Student 1 takes Quiz 1
(1, 20201088781, 79.50), -- Student 2 takes Quiz 1
(1, 20201110296, 68.00), -- Student 3 takes Quiz 1
(1, 20201448026, 74.50), -- Student 4 takes Quiz 1

-- Course 2 (Mathematics 101) with Quiz 3
(2, 20201010791, 89.00), -- Student 1 takes Quiz 2
(2, 20201088781, 77.00), -- Student 2 takes Quiz 2
(2, 20201448026, 78.75), -- Student 4 takes Quiz 2

-- Course 3 (Physics 101) with Quiz 3
(3, 20201010791, 90.50), -- Student 1 takes Quiz 3
(3, 20201088781, 87.25), -- Student 2 takes Quiz 3
(3, 20201110296, 85.00), -- Student 3 takes Quiz 3
(3, 20201448026, 80.00), -- Student 4 takes Quiz 3

-- Course 4 (Chemistry 101) with Quiz 4
(4, 20201010791, 85.00), -- Student 1 takes Quiz 4
(4, 20201088781, 80.00), -- Student 2 takes Quiz 4
(4, 20201448026, 82.50); -- Student 4 takes Quiz 4
