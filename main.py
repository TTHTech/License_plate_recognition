import RotatedImg
import DetectLicense
import Preprocess
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog, messagebox
import tkinter as tk
from PIL import Image, ImageTk
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import EasyOCR
import easyocr

reader = easyocr.Reader(['en'])

current_dir = os.path.dirname(os.path.abspath(__file__))# + '\\valid\\images\\'
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO(current_dir + '\\runs\\detect\\train10\\\weights\\best.pt')

root = Tk()
root.title("SUBJECT IMAGE PROCESSING")
root.geometry("1400x750+80+10")
root.resizable(False, False)
root.configure(bg="#2f4166")

input_path = None
current_rotated_img = None
current_photo_rotated_img = None
current_img = None
current_photoImage = None
current_ImageDetect = None
current_ImageOCR = None
current_text_license = ""
def openimage():
    global input_path, current_img, current_photoImage, input_path, current_ImageDetect
    input_path = filedialog.askopenfilename(title="Open Image",
            filetype=(("PNG file", ".png"), ("JPG file",".jpg"), ("ALL file",".")))
    if input_path:
        current_img = Image.open(input_path)
        current_ImageDetect = current_img
        width, height = current_img.size
        print(width,height)
        img = current_img.resize((f.winfo_width(), f.winfo_height())) # thay đổi kích thước ảnh thành 600x600

        current_photoImage = ImageTk.PhotoImage(img)
        lbl.configure(image=current_photoImage)
        lbl.image = current_photoImage

def showimage():
    global current_photoImage, current_img, current_ImageDetect, current_rotated_img,current_photo_rotated_img
    imgDetect = DetectLicense.DetectLicense(current_img,model)
    img = imgDetect.resize((frame2.winfo_width(), frame2.winfo_height())) # thay đổi kích thước ảnh thành 200x200
    current_ImageDetect = img
    # print(current_ImageDetect)
    current_photoImage = ImageTk.PhotoImage(current_ImageDetect)
    if current_photoImage:
        f.lift()
        lbl2.configure(image=current_photoImage)
        lbl2.image = current_photoImage
        current_rotated_img = RotatedImg.PILImgRotated(current_ImageDetect)
        current_photo_rotated_img = ImageTk.PhotoImage(current_rotated_img)
        # current_ImageDetect = current_rotated_img
        lbl4.configure(image=current_photo_rotated_img)
        lbl4.image = current_photo_rotated_img
        print(current_photoImage,current_photo_rotated_img)
    else:
        messagebox.showerror("Error", "No image to show!")

def removeimage():
    global input_path, current_img
    if input_path:
        os.remove(input_path)
        input_path = None
        current_img = None
        lbl.configure(image='', width=250, height=250)
    else:
        messagebox.showerror("Error", "No image to remove!")

def OCR():
    global current_ImageDetect, current_ImageOCR
    
    current_ImageOCR, current_text_license  = EasyOCR.OCRRESULT(current_ImageDetect,reader)
    img = current_ImageOCR.resize((frame2.winfo_width(), frame2.winfo_height())) # thay đổi kích thước ảnh thành 200x200
    # img = Preprocess.preprocess(img)
    imgPhotoOCR = ImageTk.PhotoImage(img)
    if imgPhotoOCR:
        f.lift()
        lbl3.configure(image = imgPhotoOCR)
        lbl3.image = imgPhotoOCR
        print(current_text_license)
        lbl5.config(text="Biển số: " + current_text_license)
    else:
        messagebox.showerror("Error", "No image to show!")
#icon
# image_icon = PhotoImage(file=current_dir + "\\logos\\logo.jpg")
# root.iconphoto(False, image_icon)


#logo
logo = PhotoImage(file = current_dir+ "\\logos\\logo.png")
Label(root, image = logo, bg = "#2f4155").place(x = 10 , y = 0)
Label(root, text = "License plate character recognition and extraction using image processing", bg = "#2d4155", fg = "white" , font = "arial 25 bold").place(x=100,y=20)
Label(root, text = "Đặng Huy Diệu_21110818", bg = "#2d4155", fg = "white" , font = "arial 15 bold").place(x=200,y=80)
Label(root, text = "Từ Thanh Hoài_21110826", bg = "#2d4155", fg = "white" , font = "arial 15 bold").place(x=600,y=80)
Label(root, text = "Nguyễn Văn Hoàng_21110828", bg = "#2d4155", fg = "white" , font = "arial 15 bold").place(x=1000,y=80)
#first frame
f = Frame(root, bd = 3 , bg = "black" , width=640 , height=500 , relief=GROOVE)
f.place(x=10 , y = 120)

lbl = Label(f,bg="black")
lbl.place(x=-1,y=-1)

#second frame
frame2 = Frame(root,bd=3,width=300,height=200,bg="white", relief= GROOVE)
frame2.place(x=710,y=120)

lbl2 = Label(frame2,bg="white")
lbl2.place(x=-1,y=-1)
#secondp-1 frame license plates
frame2 = Frame(root,bd=3,width= 300,height=200,bg="white", relief= GROOVE)
frame2.place(x=710,y=420)
lbl5 = Label(frame2,bg="white", text="Biển số: ")
lbl5.place(x=-1,y=-1)
############################
frame2 = Frame(root,bd=3,width= 300,height=200,bg="white", relief= GROOVE)
frame2.place(x=1080,y=120)
lbl3 = Label(frame2,bg="white")
lbl3.place(x=-1,y=-1)
#secondp-1 frame license plates
frame2 = Frame(root,bd=3,width= 300,height=200,bg="white", relief= GROOVE)
frame2.place(x=1080,y=420)

lbl4 = Label(frame2,bg="white")
lbl4.place(x=-1,y=-1)
# lbl2 = Label(frame2,bg="white")
# lbl2.place(x=40,y=10)
#third Frame
frame3 = Frame(root,bd=3,width=670,height=100,bg="#2f4155", relief= GROOVE)
frame3.place(x=10,y=640)

Button(frame3,text="Open" , width=10, height=2, font="arial 14 bold", command = openimage).place(x=100,y=30)
Button(frame3,text="Remove" , width=10, height=2, font="arial 14 bold", command = removeimage).place(x=450,y=30)
Label(frame3 , text="Picture, Image , Photo File" , bg = "#2f4155" , fg = "yellow").place(x=100,y=5)
#four Frame
frame4 = Frame(root,bd=3,width=670,height=100,bg="#2f4155", relief= GROOVE)
frame4.place(x=710,y=640)

Button(frame4,text="Recogize" , width=10, height=2, font="arial 14 bold", command = showimage).place(x=100,y=30)
Button(frame4,text="OCR" , width=10, height=2, font="arial 14 bold", command= OCR).place(x=450,y=30)
Label(frame4 , text="Picture, Image , Photo File" , bg = "#2f4155" , fg = "yellow").place(x=100,y=5)



root.resizable(False, False)
root.mainloop()

