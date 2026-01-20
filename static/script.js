


document.addEventListener("DOMContentLoaded", function() {
    function handleStudentAnswer(event) {
        const file = event.target.files[0];
        // Handle file upload and processing
        console.log("Student Answer Uploaded:", file);
    }

    function toggleModelAnswerInput() {
        const modelAnswerInput1 = document.getElementById("modelAnswerInput1");
        const modelAnswerInput2 = document.getElementById("modelAnswerInput2");
        const modelAnswerPdf = document.getElementById("modelAnswerPdf");
        const modelAnswerManual = document.getElementById("modelAnswerManual");

        if (modelAnswerManual.checked) {
            modelAnswerInput1.style.display = "none";
            modelAnswerInput2.style.display = "block";

        } else if (modelAnswerPdf.checked) {
            modelAnswerInput1.style.display = "block";
            modelAnswerInput2.style.display = "none";        }
    }

    // function calculateGrade() {
    //     // Perform grading calculation
    //     const grade = Math.floor(Math.random() * 5) + 1; // Generating a random grade between 1 and 5
    //     document.getElementById("gradeResult").innerText = `Your Grade is: ${grade}`;
    // }

    document.getElementById("modelAnswerManual").addEventListener("change", toggleModelAnswerInput);
    document.getElementById("modelAnswerPdf").addEventListener("change", toggleModelAnswerInput);
});


function duplicateSection() {
    // Clone the container element
    var container = document.querySelector('.container');
    var clone = container.cloneNode(true);
    
    // Append the clone after the original container
    container.parentNode.insertBefore(clone, container.nextSibling);
}
