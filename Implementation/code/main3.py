import tensorflow.keras
from PIL import Image, ImageOps,ImageTk
import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilename
import cv2
import imutils
import pyttsx3

classes = {
            1:'No Pothhole',
            2:'Pothhole Detected',
           }
#Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# One time initialization
engine = pyttsx3.init()

# Set properties _before_ you add things to say
engine.setProperty('rate', 125)    # Speed percent (can go over 100)
engine.setProperty('volume', 1)  # Volume 0-1

#initialise GUI
root = tk.Tk()
root.title("Pothhole_detection")

root.geometry("600x550")
root.configure(background ="white")
title = tk.Label(text="Select An Image To Process", background = "white", fg="Brown", font=("", 15))
title.grid(row=0, column=2, padx=10, pady = 10)

def exit():
        root.destroy()

def clear():
    string.destroy()
        
def analysis():
    global string
        # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(path)

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    #image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    idx = np.argmax(prediction)
    final_prediction = classes[idx+1]

   
    engine.say(final_prediction)
    # Flush the say() queue and play the audio
    engine.runAndWait()



    string = tk.Label(text = final_prediction, background="white",fg="Black", font=("", 15))
    string.grid(column=4, row=3, padx=10, pady=10)
    button3 = tk.Button(text="Clear", command=clear)
    button3.grid(row=6, column=2, padx=10, pady = 10)
    button4 = tk.Button(text="Exit", command=exit)
    button4.grid(row=7, column=2, padx=10, pady = 10)
   
def video_input():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            cv2.imwrite("img.jpg",frame)
            frame = imutils.resize(frame,width=400)
            fin = frame.copy()
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

            # Replace this with the path to your image
            image = Image.open("img.jpg")

            #resize the image to a 224x224 with the same strategy as in TM2:
            #resizing the image to be at least 224x224 and then cropping from the center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)

            #turn the image into a numpy array
            image_array = np.asarray(image)

            # display the resized image
            #image.show()

            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

            # Load the image into the array
            data[0] = normalized_image_array

            # run the inference
            prediction = model.predict(data)
            idx = np.argmax(prediction)
            final_prediction = classes[idx+1]
            cv2.putText(fin, final_prediction, (10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,0), 2)
            cv2.imshow('image', fin)
            engine.say(final_prediction)
            # Flush the say() queue and play the audio
            engine.runAndWait()
            k = cv2.waitKey(1) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                    break
                
        
def openphoto():
    global path
    path=askopenfilename(filetypes=[("Image File",'.jpg')])
    frame = cv2.imread(path)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cv2image = imutils.resize(cv2image, width=250)
    img = Image.fromarray(cv2image)
    tkimage = ImageTk.PhotoImage(img)
    myvar=tk.Label(root,image = tkimage, height="250", width="250")
    myvar.image = tkimage
    myvar.place(x=1, y=0)
    myvar.grid(row=3, column=2 , padx=10, pady = 10)
    button2 = tk.Button(text="Analyse", command=analysis)
    button2.grid(row=4, column=2, padx=10, pady = 10)
    

button1 = tk.Button(text="Choose Image", command = openphoto)
button1.grid(row=1, column=2, padx=10, pady = 10)

button1 = tk.Button(text="Video", command = video_input)
button1.grid(row=2, column=2, padx=10, pady = 10)

root.mainloop()
