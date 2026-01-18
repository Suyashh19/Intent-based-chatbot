# Intent-Based Chatbot 🤖

A simple **intent-based chatbot** built using **Python and Scikit-learn**.
The chatbot classifies user input into predefined intents using **TF-IDF vectorization** and a **machine learning classifier**, displaying the predicted intent through a web interface.

🔗 **Live Demo:** [https://intent-based-chatbot-inxg.onrender.com/](https://intent-based-chatbot-inxg.onrender.com/)

---

## 📌 Features

- **Intent Classification:** Uses Machine Learning to categorize text.
- **TF-IDF Vectorization:** Converts text inputs into meaningful numerical representations.
- **Web Interface:** Built with HTML, CSS, and JavaScript for user interaction.
- **Flask Backend:** Serves predictions via a REST API.
- **Cloud Deployment:** Hosted online using Render.

---

## 🛠️ Tech Stack

- **Language:** Python
- **ML Library:** Scikit-learn
- **Vectorization:** TF-IDF
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Render

---

## 🧠 How It Works

1. **Input:** User enters a message in the web interface.
2. **Process:** The input is sent to the Flask backend (`/predict` endpoint).
3. **Prediction:** The trained ML model predicts the most likely intent based on the training data.
4. **Output:** The predicted intent is displayed on the UI.

> **Note:** This is a **closed-domain, intent-recognition system**. It is designed to classify specific intents, not to hold open-ended free conversations.

---

## 📂 Project Structure

```text
Intent-based-chatbot/
├── app.py              # Main Flask application
├── model.pkl           # Trained classifier model
├── vectorizer.pkl      # Saved TF-IDF vectorizer
├── train.py            # Script to train the model
├── requirements.txt    # Python dependencies
├── Procfile            # Render deployment configuration
├── templates/
│   └── index.html      # Frontend user interface
└── data/
    └── intents.json    # Dataset of intents and patterns
🚀 Running Locally
Follow these steps to run the project on your machine.

1. Clone the repository

Bash

git clone [https://github.com/Suyashh19/Intent-based-chatbot.git](https://github.com/Suyashh19/Intent-based-chatbot.git)
cd Intent-based-chatbot
2. Install dependencies

Bash

pip install -r requirements.txt
3. Run the application

Bash

python app.py
4. Access the Chatbot Open your browser and navigate to:

Plaintext

[http://127.0.0.1:5000](http://127.0.0.1:5000)
📊 Model Performance
Accuracy: Achieved ~92% accuracy on the custom intent dataset.

Evaluation: Tested using a standard train-test split and classification metrics.

📌 Notes
The chatbot is designed to always predict the closest known intent.

Casual or out-of-scope inputs will be mapped to the nearest mathematical probability in the dataset.

This project is designed primarily for learning and demonstration purposes.

👤 Author
Suyash Patil

📜 License
This project is created for educational purposes.


### Next Step
Would you like me to write the `requirements.txt` file content based on the libraries mentione
