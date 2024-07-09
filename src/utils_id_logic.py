import random

def generate_random_number():
  """Generates a random 12-digit number as a string."""
  while True:
    number = str(random.randint(10**11, 10**12 - 1))
    if len(number) == 12:
      return number
def calculate_average_digit_distance(num1, num2):
  """Calculates the average absolute difference between corresponding digits of two numbers."""
  total_distance = 0
  for i in range(12):
    digit1 = int(num1[i])
    digit2 = int(num2[i])
    total_distance += abs(digit1 - digit2)
  return total_distance / 12
def ID_generator(IDS):
    exam_ids ={}
    for id in IDS:
        exam_ids[id] = generate_random_number()
    return(exam_ids)
# Function to find the key based on value
def get_key(value, my_dict):
  for key, val in my_dict.items():
    if val == value:
      return key
  return None  # If value not found
import numpy as np
def checkingg(student_id,id_dict):
    all_ids =list(id_dict.values())
    average_distances = []
    if student_id in all_ids:
        print("Right Exam ID your university ID is: ", get_key(student_id,id_dict))
    else:
        for id in all_ids:
            average_distances.append(calculate_average_digit_distance(student_id, id))
        pos = np.array(average_distances).argmin()
        print("Worng ID but your university ID is most likey: ", get_key(all_ids[pos],id_dict))
        
        
    
    