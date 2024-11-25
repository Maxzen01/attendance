from flask import Flask, request, render_template, redirect, url_for
import os
import cv2
import joblib
import pandas as pd
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

nimgs = 10
imgBackground = cv2.imread("background.png")
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
attendance_status = {}

# Ensure required directories and files exist
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance_Login_{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance_Login_{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time\n')
if f'Attendance_Logout_{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance_Logout_{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time\n')

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
    return face_points

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces, labels = [], []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(np.array(faces), labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance(file_name):
    df = pd.read_csv(file_name)
    return df['Name'], df['Roll'], df['Time'], len(df)

def has_attendance(name, action):
    """Check if the user already has login/logout attendance for the day."""
    filename = f'Attendance/Attendance_{"Login" if action == "login" else "Logout"}_{datetoday}.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        return name in df['Name'].values
    return False

def add_attendance(name, action):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    file_name = f'Attendance/Attendance_{"Login" if action == "login" else "Logout"}_{datetoday}.csv'

    if has_attendance(name, action):
        print(f"{username} has already marked {action} today!")
        return  # Skip adding attendance if it already exists

    with open(file_name, 'a') as f:
        f.write(f'{username},{userid},{current_time}\n')
    print(f"{username} has been marked for {action} at {current_time}")

def getallusers():
    userlist = os.listdir('static/faces')
    names, rolls = zip(*[user.split('_') for user in userlist])
    return userlist, names, rolls, len(userlist)

@app.route('/')
def home():
    # Extract login and logout data for display
    login_names, login_rolls, login_times, login_count = extract_attendance(f'Attendance/Attendance_Login_{datetoday}.csv')
    logout_names, logout_rolls, logout_times, logout_count = extract_attendance(f'Attendance/Attendance_Logout_{datetoday}.csv')
    return render_template('home.html', login_names=login_names, login_rolls=login_rolls, login_times=login_times, 
                           login_count=login_count, logout_names=logout_names, logout_rolls=logout_rolls, 
                           logout_times=logout_times, logout_count=logout_count, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/start', methods=['GET', 'POST'])
def start():
    action = request.args.get("status", "login")  # Capture login or logout status
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', mess='No trained model found. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    face_detected = False
    identified_person = None
    success_message_displayed = False

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            faces = extract_faces(frame)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]
                
                # Show identified name in red while detecting
                cv2.putText(frame, f'{identified_person}', (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                face_detected = True

            # Display video feed
            imgBackground[162:162 + 480, 55:55 + 640] = frame
            cv2.imshow('Attendance', imgBackground)

        # Press Enter to mark attendance if face is correctly detected
        if face_detected and cv2.waitKey(1) == 13:  # 'Enter' key pressed
            if identified_person:
                add_attendance(identified_person, action)
                
                # Show success message
                success_message_displayed = True
                face_detected = False  # Reset the detection state after marking attendance

        if success_message_displayed:
            # Clear the frame for the success message display
            frame[:] = (255, 255, 255)
            cv2.putText(frame, "Attendance successfully taken", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Attendance", frame)
            cv2.waitKey(1000)  # Display the message for a moment
            success_message_displayed = False  # Reset success message display flag

        if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for('home'))

@app.route('/add', methods=['POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{str(newuserid)}'

    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while cap.isOpened():
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            if j % 5 == 0:
                cv2.imwrite(f'{userimagefolder}/{newusername}_{i}.jpg', frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if i >= nimgs:
            break
        cv2.imshow('Capturing Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    train_model()
    return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2, mess="New face has been added!")

if __name__ == '__main__':
    app.run(debug=True)
