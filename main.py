import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# get csv files
path = r'scifi_books.csv'
# read csv files into dataframe
df = pd.read_csv(path, header=0, encoding='unicode_escape')
# drop duplicates
df = df.drop_duplicates()

# book ratings histogram
ratings_hist = df['Rating_score'].hist(grid=False, bins=10)
ratings_hist.set_xlabel('Book Ratings')
ratings_hist.set_ylabel('Number of User Ratings')
plt.show()

# cosine similarity
count_matrix = CountVectorizer().fit_transform(df['Genres'])
similarity = cosine_similarity(count_matrix)

# get book title from user
book_user_likes = input('What is your favorite Science Fiction book?')
matched_books = df[df['Book_Title'].str.contains(book_user_likes, case=False)]
# get first book that matches
book_title = matched_books['Book_Title'].values[0]
print('Searching for book recommendations based on: ' + book_title)

# get ID of book user likes
book_id_list = np.where(df['Book_Title'] == book_title)
book_id = int(book_id_list[0])

# list of all books and similarity scores
scores = list(enumerate(similarity[book_id]))

# sort books by similarity
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
# first book is the one the user picked, remove from list
sorted_scores = sorted_scores[1:]

# get top 10 most similar books and their scores
top_ten = []
j = 0
for item in sorted_scores:
    top_ten.append(item)
    j = j + 1
    if j >= 10:
        break

top_ten_titles = []
top_ten_scores = []
for item in top_ten:
    row = df.iloc[item[0]]
#    top_ten_titles.append(df[df.index == item[0]]['Book_Title'].values[0])
    top_ten_titles.append(row['Book_Title'])
    top_ten_scores.append(item[1])

recommended_books = [(top_ten_titles[i], top_ten_scores[i]) for i in range(0, len(top_ten_titles))]

# recommend top 10 most similar books to user
print('Your Top 10 most recommended books, based on ' + book_title + ':\n')
i = 0
for item in top_ten_titles:
    print(str(i) + ' ' + item)
    i = i + 1

# pie chart of top 10 and similarity
figureObject, axesObject = plt.subplots()
axesObject.pie(top_ten_scores, labels=top_ten_titles, autopct='%0.3f', startangle=90)
axesObject.axis('equal')
plt.show()

# histogram of recommended books and similarity
plt.bar(top_ten_titles, top_ten_scores, color=['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'red', 'orange',
                                               'yellow', 'green'])
plt.xlabel('Top 10 Recommended Books')
plt.ylabel('Similarity to Your Book')
plt.xticks(rotation=90)
plt.show()


