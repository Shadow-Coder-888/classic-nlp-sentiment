import joblib

model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

while True:
    text = input("Enter text: ").strip()

    if text.lower() == "exit":
        break

    if not text:
        print("Empty input ignored")
        continue

    X = vectorizer.transform([text])
    prediction = model.predict(X)

    print("Prediction:", prediction[0])
