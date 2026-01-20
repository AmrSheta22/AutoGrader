# List all professors and their courses
SELECT 
    p.name AS professor_name, 
    c.course_name 
FROM 
    Professors p
JOIN 
    Courses c ON p.professor_id = c.professor_id;
# List all students enrolled in each course
SELECT 
    c.course_name, 
    s.name AS student_name 
FROM 
    Enrollments e
JOIN 
    Courses c ON e.course_id = c.course_id
JOIN 
    Students s ON e.student_id = s.student_id
ORDER BY 
    c.course_name, s.name;
#Get the average quiz score for each student
SELECT 
    s.name AS student_name, 
    AVG(qr.score) AS average_score 
FROM 
    QuizResults qr
JOIN 
    Students s ON qr.student_id = s.student_id
GROUP BY 
    s.name;
# Get the average quiz score for each course
SELECT 
    c.course_name, 
    AVG(qr.score) AS average_score 
FROM 
    QuizResults qr
JOIN 
    Quizzes q ON qr.quiz_id = q.quiz_id
JOIN 
    Courses c ON q.course_id = c.course_id
GROUP BY 
    c.course_name;
#Get the highest and lowest score in each quiz
SELECT 
    q.quiz_name, 
    MAX(qr.score) AS highest_score, 
    MIN(qr.score) AS lowest_score 
FROM 
    QuizResults qr
JOIN 
    Quizzes q ON qr.quiz_id = q.quiz_id
GROUP BY 
    q.quiz_name;
# List all quizzes with their respective courses and professors
SELECT 
    q.quiz_name, 
    q.quiz_date, 
    q.pdf_path, 
    c.course_name, 
    p.name AS professor_name 
FROM 
    Quizzes q
JOIN 
    Courses c ON q.course_id = c.course_id
JOIN 
    Professors p ON c.professor_id = p.professor_id
ORDER BY 
    q.quiz_date;
#Find the number of students enrolled in each course
SELECT 
    c.course_name, 
    COUNT(e.student_id) AS number_of_students 
FROM 
    Enrollments e
JOIN 
    Courses c ON e.course_id = c.course_id
GROUP BY 
    c.course_name;
#Get the quiz results for a specific student
SELECT 
    s.name AS student_name, 
    c.course_name, 
    q.quiz_name, 
    qr.score 
FROM 
    QuizResults qr
JOIN 
    Students s ON qr.student_id = s.student_id
JOIN 
    Quizzes q ON qr.quiz_id = q.quiz_id
JOIN 
    Courses c ON q.course_id = c.course_id
WHERE 
    s.name = 'Anna Wilson'; -- Change the student name as needed
#Find all students who scored above a certain threshold in a specific quiz
SELECT 
    s.name AS student_name, 
    q.quiz_name, 
    qr.score 
FROM 
    QuizResults qr
JOIN 
    Students s ON qr.student_id = s.student_id
JOIN 
    Quizzes q ON qr.quiz_id = q.quiz_id
WHERE 
    q.quiz_name = 'Final Exam' -- Change the quiz name as needed
    AND qr.score > 80; -- Change the score threshold as needed

# List students and their total quiz scores in each course
SELECT 
    s.name AS student_name, 
    c.course_name, 
    SUM(qr.score) AS total_score 
FROM 
    QuizResults qr
JOIN 
    Students s ON qr.student_id = s.student_id
JOIN 
    Quizzes q ON qr.quiz_id = q.quiz_id
JOIN 
    Courses c ON q.course_id = c.course_id
GROUP BY 
    s.name, c.course_name;

