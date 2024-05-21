# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:40:07 2024

"""

import re
import pandas as pd
import os

def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Function to parse the content of a file and extract data
def parse_file_content(content):
    car_name_match = re.search(r'<DOCNO>(.*?)</DOCNO>', content)
    if car_name_match:
        car_name = car_name_match.group(1)
    else:
        car_name = None
    
    comments = []
    comment_matches = re.findall(r'<DOC>.*?<DATE>(.*?)</DATE>.*?<AUTHOR>(.*?)</AUTHOR>.*?<TEXT>(.*?)</TEXT>.*?<FAVORITE>(.*?)</FAVORITE>.*?</DOC>', content, re.DOTALL)
    for date, author, text, favorite in comment_matches:
        full_text = text.strip()
        if favorite.strip():
            full_text += " " + favorite.strip()
        comments.append({'Date': date.strip(), 'Author': author.strip(), 'Text': full_text})
    
    return car_name, comments

# Function to read all files in a directory and extract content
def read_files(directory):
    data = []
    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    content = file.read()
                    car_name, comments = parse_file_content(content)
                    if car_name:
                        data.extend([{'Car Name': car_name, **comment} for comment in comments])
    return data


if __name__ == "__main__":
    # Read all files in a directory and store data in a DataFrame
    directory = './OpinRank/cars/data'
    all_comments = read_files(directory)
    df = pd.DataFrame(all_comments)
    df['cleaned_text'] = df['Text'].apply(clean_text)
    df.to_excel("data.xlsx")
