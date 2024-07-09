import os
import pandas as pd

if __name__ == "__main__":
    df = pd.DataFrame(
        {"question": [], "model_answer": [], "student_answer": [], "grade": []}
    )
    # to do: add all csv files
    all_data = [
        "duorcv2.csv",
        "qa_7500.csv.csv",
        "short_answer_grading.csv",
        "SQUAD Dataset.csv",
        "Tensorflow Dataset.csv",
        "wikiqa.csv",
        "finished_train_v2.1_file.csv",
        "finished_ori_pqaa_file.csv",
        "finished_ori_pqau_file.csv",
        "finished_dev_v2.1_file.csv",
        "finished_ori_pqal_file.csv",
    ]
    for i in os.listdir("."):
        # concatinate all csv files according to the column names
        if i.endswith(".csv"):
            print(i)
            df_i = pd.read_csv(i)
            if (
                not ("model_answer" in df_i.columns)
                or df_i["model_answer"].dtype == "float64"
            ):
                df_i["model_answer"] = [" "] * len(df_i)
            if (
                not ("student_answer" in df_i.columns)
                or df_i["student_answer"].dtype == "float64"
            ):
                df_i["student_answer"] = [" "] * len(df_i)
            df_i["source"] = [i] * len(df_i)
            df = pd.concat([df, df_i], ignore_index=True)
    df.to_csv("all.csv", index=False)
