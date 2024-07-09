import argparse
import os
import pandas as pd
from utils_spelling_correction import *

parser = argparse.ArgumentParser(description="Spell check data")
parser.add_argument(
    "--input",
    type=str,
    help="Input file path",
    default="all.csv",
)

if __name__ == "__main__":
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    sym_spell = SpellChecker()
    corrected_question = []
    corrected_model_answer = []
    corrected_student_answer = []

    import multiprocessing as mp
    from multiprocessing import Pool
    with Pool(mp.cpu_count()) as p:
        corrected_question = p.map(correct_text, df['question'])
        print("Question done")
        corrected_model_answer = p.map(correct_text, df['model_answer'])
        print("Model answer done")
        corrected_student_answer = p.map(correct_text, df['student_answer'])
        print("Student answer done")
        

    df['question'] = corrected_question
    df['student_answer'] = corrected_student_answer
    df.to_csv('all_corrected.csv', index=False)