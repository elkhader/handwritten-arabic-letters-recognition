import os
from PIL import ImageTk, Image, ImageDraw
import PIL
import pickle
from tkinter import *
import numpy as np
from skimage.feature import local_binary_pattern


width = 200  # canvas width
height = 200 # canvas height
center = height//2
white = (255, 255, 255) # canvas back

ar_letters_rom =["alif","bāʼ","tāʼ","thāʼ","jīm","ḥāʼ","khāʼ","dāl","dhāl","rāʼ","zayn/zāy","sīn","shīn","ṣād","ḍād","ṭāʼ","ẓāʼ","ʻayn","ghayn","fāʼ","qāf","kāf","lām","mīm","nūn","hāʼ","wāw","yāʼ"]
ar_letters =["أ","ب","ت","ث","ج","ح","خ","د","ذ","ر","ز","س","ش","ص","ض","ط","ظ","ع","غ","ف","ق","ك","ل","م","ن","ه","و","ي"]
ar_letters_whole =["ألف","باء","تاء","ثاء","جيم","حاء","خاء","دال","ذال","راء","زاي","سين","شين","صاد","ضاد","طاء","ظاء","عين","غين","فاء","قاف","كاف","لام","ميم","نون","هاء","واو","ياء"]


#getting model
cwd = os.getcwd()
cwd=cwd.replace("application", "model\\knn_model.pkl")
with open(cwd, 'rb') as file:
    knn_model = pickle.load(file)

#LBP fucntion
def apply_LBP(arr):
    arr_reshaped=np.array(np.reshape(arr, (32, 32)), dtype="uint8")
    LBPed=local_binary_pattern(arr_reshaped,3,8,method='default')
    LBPed=np.array(np.reshape(LBPed, (1, 1024)), dtype="uint8")
    return LBPed


def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black",width=10)
    draw.line([x1, y1, x2, y2],fill="black",width=15)

def update():
    txt.insert(END,"Changed \n hello")


def predict():
    from PIL import Image
    image= output_image.transpose(Image.FLIP_LEFT_RIGHT)
    image= image.rotate(90)
    image= image.resize((32,32))
    numpydata=np.asarray(image)
    numpydata=np.reshape(numpydata,(1,1024))
    LBP=apply_LBP(numpydata)
    prediction=knn_model.predict(LBP)
    prediction_text= str("\n" + ar_letters[prediction[0]-1]+ " - "+ar_letters_rom[prediction[0]-1]+" - "+ ar_letters_whole[prediction[0]-1])
    txt.insert(END, prediction_text)
    #print(prediction[0])


def clear():
    from PIL import Image,ImageDraw
    global output_image
    global draw
    txt.delete(1.0,END)
    output_image = PIL.Image.new("P", (200, 200), white)
    draw = ImageDraw.Draw(output_image)
    canvas.delete('all')


master = Tk()

# create a tkinter canvas to draw onq
canvas = Canvas(master, width=width, height=height, bg='white')
canvas.pack()

# create an empty PIL image and draw object to draw on
output_image = PIL.Image.new("P", (200, 200), white)
draw = ImageDraw.Draw(output_image)

txt=Text(master,bd=2,exportselection=0,bg='WHITE',font=('Helvetica', 18, 'bold'),height=3,width=15,padx=5, pady=70)
txt.tag_configure("center", justify='center')



canvas.pack(expand=YES, fill=BOTH,side=LEFT)
canvas.bind("<B1-Motion>", paint)

txt.pack(side=RIGHT, expand=YES, fill=BOTH)
txt.tag_configure("center", justify='center')
#txt.insert(END,"\n")


button=Button(text=" Predict  ",command=predict)
button.pack(side=LEFT)

button2=Button(text="   Clear   ",command=clear)
#button2.pack()
button2.place(x=204, y=135)

master.mainloop()    
