import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords', quiet=True)

# Built-in small dataset - trains in seconds!
data = {
    'message': [
        # Phishing emails
        "Your account has been suspended. Click here to verify your details immediately or lose access.",
        "Congratulations! You won a prize. Enter your bank details to claim your reward now.",
        "URGENT: Your PayPal account is limited. Verify now at http://paypal-secure.tk/login",
        "Dear customer your bank account will be closed. Update your information immediately.",
        "You have a pending payment. Click this link to confirm your account details.",
        "Your Apple ID has been locked. Verify your identity at http://apple-id-verify.com",
        "Win an iPhone! Click here to claim your free prize. Limited time offer!",
        "Your Netflix account will be suspended. Update payment at http://netflix-billing.tk",
        "ALERT: Unauthorized login detected. Secure your account now by clicking here.",
        "Your Amazon account is on hold. Confirm your details to restore access immediately.",
        "Verify your email now or your account will be deleted within 24 hours.",
        "You have been selected for a cash reward. Provide your details to receive $1000.",
        "Your password has expired. Click here to reset it immediately.",
        "Warning: Your account shows suspicious activity. Verify now to avoid suspension.",
        "Claim your lottery winnings! Send your bank details to receive your prize money.",
        "Your credit card has been charged. Dispute this transaction by clicking here.",
        "Important security update required. Login to verify your account details now.",
        "Your insurance policy is due. Click here to renew and avoid cancellation.",
        "Tax refund pending. Submit your information to receive your government refund.",
        "Final warning: Your account will be terminated unless you verify your identity.",

        # Safe emails
        "Hi team, please find the meeting notes attached. Let me know if you have questions.",
        "Hey, are we still on for lunch tomorrow? Let me know what time works for you.",
        "Please review the attached report and share your feedback by end of this week.",
        "The project deadline has been extended to next Friday. Update your tasks accordingly.",
        "Happy birthday! Hope you have a wonderful day filled with joy and celebration.",
        "Reminder: Team meeting tomorrow at 10am in the conference room. Please be on time.",
        "Thank you for your application. We will review it and get back to you shortly.",
        "The quarterly report is ready. Please find it attached for your review.",
        "Just checking in to see how the new project is going. Let me know if you need help.",
        "I wanted to share this interesting article I found about machine learning trends.",
        "Can you send me the updated schedule for next month? Thanks in advance.",
        "Great work on the presentation today! The client was really impressed.",
        "Reminder to submit your timesheet by end of day Friday.",
        "The office will be closed on Monday for the public holiday.",
        "Please welcome our new team member joining us next week.",
        "Your order has been shipped and will arrive in 3-5 business days.",
        "Here are the minutes from yesterday's meeting for your reference.",
        "Looking forward to seeing everyone at the annual company picnic this Saturday.",
        "Could you please review this document and suggest any improvements?",
        "The training session has been rescheduled to next Tuesday at 2pm.",
    ],
    'label': [1]*20 + [0]*20  # 1=phishing, 0=safe
}

df = pd.DataFrame(data)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([w for w in text.split() if w not in stopwords.words('english')])
    return text

df['message'] = df['message'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['message'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

with open('pickle/email_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('pickle/email_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Email model trained and saved successfully!")
