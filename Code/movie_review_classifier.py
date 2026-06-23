from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample data - expanded dataset
reviews = [
    # Positive reviews
    "This movie was fantastic and thrilling",
    "I loved every moment of this film",
    "Amazing storyline and great acting",
    "One of the best movies I have ever seen",
    "Wonderful film with brilliant performances",
    "A masterpiece of modern cinema truly breathtaking",
    "The acting was superb and the plot was gripping",
    "Highly recommend this film to everyone",
    "Beautiful cinematography and outstanding direction",
    "This film left me speechless absolutely incredible",
    "Great characters and an engaging story throughout",
    "I was on the edge of my seat the whole time",
    "A feel good movie that everyone will enjoy",
    "The best film I have watched this year",
    "Incredible performance by the lead actor loved it",
    "Heartwarming story with a perfect ending",
    "Brilliant direction and a captivating screenplay",
    "This movie exceeded all my expectations",
    "A stunning visual experience with a great story",
    "Loved the chemistry between the characters amazing",

    # Negative reviews
    "This movie was terrible and boring",
    "I hated this film it was awful",
    "Worst movie ever complete waste of time",
    "Terrible acting and poor storyline",
    "I did not enjoy this movie at all",
    "Dull and uninteresting film avoid at all costs",
    "The plot made no sense whatsoever disappointing",
    "Bad directing and the acting was cringe worthy",
    "I fell asleep halfway through so boring",
    "A complete disaster from start to finish",
    "Poorly written script with no character development",
    "One of the worst films I have ever seen",
    "The story was confusing and the ending was terrible",
    "Waste of money and time do not watch this",
    "Horrible movie with no redeeming qualities at all",
    "The acting was wooden and the plot was predictable",
    "Absolutely dreadful film I want my money back",
    "Nothing made sense and the pacing was awful",
    "Boring characters and a painfully slow story",
    "Deeply disappointing and a total waste of talent",
]

labels = [1]*20 + [0]*20  # 1 = positive, 0 = negative

# Preprocess and train
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Predictions:", ["Positive" if p == 1 else "Negative" for p in predictions])
print("Actual:     ", ["Positive" if l == 1 else "Negative" for l in y_test])
print(f"Accuracy:    {accuracy_score(y_test, predictions)}\n")

# Interactive input
print("=== Movie Review Classifier ===")
print("Type a movie review and press Enter to classify it.")
print("Type 'quit' to exit.\n")

while True:
    user_review = input("Enter your review: ")
    if user_review.lower() == 'quit':
        print("Goodbye!")
        break
    transformed = vectorizer.transform([user_review])
    result = model.predict(transformed)[0]
    print(f"Sentiment: {'Positive' if result == 1 else 'Negative'}\n")