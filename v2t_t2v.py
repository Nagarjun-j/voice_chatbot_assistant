import speech_recognition as sr
import pyttsx3

r = sr.Recognizer()

def voice_to_text():
    with sr.Microphone() as source:
            print("Say or ask something in the microphone")
            audio = r.listen(source)
            text = r.recognize_google(audio)
            text = text.lower()
            print(f"Recognized text is:{ text}")
voice_to_text()

def text_to_voice(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.say(text)
    engine.runAndWait()
    print("Output voice given")
    rate = engine.getProperty('rate')
    print (rate)

# text_to_voice("Hello i am nagarjun, hope you are doing good!")

"Good morning! How are you? Did you sleep well? I'll have a coffee, please"
"What's the weather like today? Have a great day!"
"Let's catch up later. Thank you so much!"
"Sorry, I'm running late. Can you please pick up milk?"


"Utilizing ensemble(and sample, and symbol) methods such as gradient boosting or random forests, "
"we aim to mitigate overfitting and enhance model generalization "
"by aggregating the predictions of multiple weak learners into a strong predictive model."




