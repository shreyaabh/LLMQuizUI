import streamlit as st
import pandas as pd
import io
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
import xlsxwriter
import os
from PIL import Image   


# Setup for LangChain and model configuration
model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    openai_api_key="k-zP8S3pdskdjksdjskdjeila99kUJ34qqT3BlbkFJ9kuAJkLwHYkKtK62G18",  # Replace with your OpenAI API key
    temperature=0,
    max_tokens=4095,
)

prompt_template = """Role: Act as a particular subject teacher and complete the given task below as shown in example.

Task: Generate a quiz with 10 questions. The quiz should contain MCQ with 4 options and only one correct answer. 

Example:
query = Generate quiz on topic 'addition up to 25' in language 'English' for subject 'Math'. Generate '1' number of questions in 'medium' difficulty"

Output Format : 

Question : What is 12 + 4?
Subject : Math
Language : English
Difficulty: Medium
Topic : addition up to 25
Answer 1 : 16
Answer 2 : 19
Answer 3 : 14
Answer 4 : 18

Most Important Instructions:
1. The most important thing is that the Correct answer is always in Answer 1. 
2. Always maintain the particular topic and output format while generating.
3. Always generate 10 questions.
{context} 
"""

prompt = PromptTemplate(
    input_variables=["context"],
    template=prompt_template,
)

chain = LLMChain(llm=model, prompt=prompt)

image = Image.open('Chimpvine.jpg')
st.image(image,width=200)
st.title("Quiz Generator - Chimpvine")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df1 = pd.read_excel(uploaded_file)
    df1 = df1.dropna()
    df1.rename(columns=lambda x: x.strip(), inplace=True)
    with open("title.txt", "w") as file:
        file.write("")
    file.close()
    text = """"""
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook("xlsx_file.xlsx")
    workbook.close()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:


        # Iterate through each row in the DataFrame and generate quiz content
        for index, row in df1.iterrows():
            topic = row["topic"]
            language = row["language"]
            subject = row["subject"]
            number = row["number"]
            difficulty = row["difficulty"]

            query = f"Generate quiz on topic '{topic}' in language '{language}' for subject '{subject}'. Generate {number} number of questions in '{difficulty}' difficulty mode"
            content = chain.invoke({"context": query})
            text += content["text"] + "\n"

            file1 = open("title.txt", "w")  # write mode
            file1.write(text + "\n")
            file1.close()

        columns = [
            "Subject",
            "Language",
            "Difficulty",
            "Question",
            "Topic",
            "Answer 1",
            "Answer 2",
            "Answer 3",
            "Answer 4",
        ]
        rows = []
        row_data = {}
        current_col = None

        with open("title.txt", "r", errors="ignore") as file:
            for line in file:
                line = line.strip()

                # column check garne
                if any(line.startswith(col) for col in columns):

                    colon_pos = line.find(":")
                    if colon_pos != -1:
                        # colon pachi ko sab haldiney
                        current_col = next(
                            col for col in columns if line.startswith(col)
                        )
                        row_data[current_col] = line[colon_pos + 1 :].strip()
                elif line and current_col in row_data:
                    # aba loop chalaune
                    colon_pos = line.find(":")
                    if colon_pos != -1:
                        row_data[current_col] += " " + line[colon_pos + 1 :].strip()
                else:
                    # reject garne if blank cha bhane
                    if row_data:
                        rows.append(row_data)
                        row_data = {}

            if row_data:
                rows.append(row_data)

        df_existing = pd.read_excel("xlsx_file.xlsx")
        df = pd.DataFrame(rows, columns=columns)

        finaldf = pd.concat(
            [df_existing, df], ignore_index=True
        )  # Placeholder for actual parsing logic
        finaldf.to_excel(writer, sheet_name="xlsx_file.xlsx", index=False)

    # The ExcelWriter is closed automatically here
    output.seek(0)

    # Provide a download button for the file
    st.download_button(
        label="Download Quiz as Excel",
        data=output,
        file_name="xlsx_file.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.write("The file will be downloaded to your default Download path.")