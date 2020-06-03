import streamlit as st
import os
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import tensorflow as tf


#---------code to that lets user select a file from the curdir----#
st.title("Face Detection Web App")
st.sidebar.title("Select your options!")
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()
st.write('You selected `%s`' % filename)
#------------------file selection code ends here----------------#

#-----code to display image before detection--------------------#
image_path = filename
# noinspection PyUnresolvedReferences
im = cv2.imread(image_path)
# noinspection PyUnresolvedReferences
img_rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

st.image(img_rgb, caption='Before Detection...')
#---------before detection ends here--------------------#


#-----------detecting faces------------------#
select_color = st.sidebar.selectbox("Color?",("Red", "Green", "Blue"))
if(select_color=="Green"):
    color=(0,255,0)
if(select_color=="Red"):
    color=(0,0,255)
if(select_color=="Blue"):
    color=(255,0,0)
thick = st.sidebar.number_input("Thickness of bounding box", 1, 6, step=1, key='thick')
select_font = st.sidebar.selectbox("Font?",("FONT_HERSHEY_SIMPLEX","FONT_HERSHEY_PLAIN","FONT_HERSHEY_DUPLEX","FONT_HERSHEY_COMPLEX","FONT_HERSHEY_TRIPLEX","FONT_HERSHEY_COMPLEX_SMALL","FONT_HERSHEY_SCRIPT_SIMPLEX","FONT_HERSHEY_SCRIPT_COMPLEX","FONT_ITALIC"))
if(select_font=="FONT_HERSHEY_SIMPLEX"):
    # noinspection PyUnresolvedReferences
    font=cv2.FONT_HERSHEY_SIMPLEX
if(select_font=="FONT_HERSHEY_PLAIN"):
    # noinspection PyUnresolvedReferences
    font=cv2.FONT_HERSHEY_PLAIN
if(select_font=="FONT_HERSHEY_DUPLEX"):
    # noinspection PyUnresolvedReferences
    font=cv2.FONT_HERSHEY_DUPLEX
if(select_font=="FONT_HERSHEY_COMPLEX"):
    # noinspection PyUnresolvedReferences
    font=cv2.FONT_HERSHEY_COMPLEX
if(select_font=="FONT_HERSHEY_TRIPLEX"):
    # noinspection PyUnresolvedReferences
    font=cv2.FONT_HERSHEY_TRIPLEX
if(select_font=="FONT_HERSHEY_COMPLEX_SMALL"):
    # noinspection PyUnresolvedReferences
    font=cv2.FONT_HERSHEY_COMPLEX_SMALL
if(select_font=="FONT_HERSHEY_SCRIPT_SIMPLEX"):
    # noinspection PyUnresolvedReferences
    font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
if(select_font=="FONT_HERSHEY_SCRIPT_COMPLEX"):
    # noinspection PyUnresolvedReferences
    font=cv2.FONT_HERSHEY_SCRIPT_COMPLEX
if(select_font=="FONT_ITALIC"):
    # noinspection PyUnresolvedReferences
    font=cv2.FONT_ITALIC
fontScale = st.sidebar.number_input("Font Scale", 0.1, 1.0, step=0.1, key='fontscale')
select_text_color = st.sidebar.selectbox("Text Color?",("Black", "White"))
if(select_text_color=="Black"):
    color_text=(0,0,0)
if(select_text_color=="White"):
    color_text=(255,255,255)
user_input = st.sidebar.text_input("Enter Text to be displayed")
thickness = st.sidebar.number_input("Thickness of text", 1, 4, step=1, key='text_thick')
if (st.sidebar.button("Detect!", key='detect')):
    faces, confidences = cv.detect_face(im)
    # loop through detected faces and add bounding box
    for face in faces:
        (startX,startY) = face[0],face[1]
        (endX,endY) = face[2],face[3]
        # draw rectangle over face
        # noinspection PyUnresolvedReferences
        cv2.rectangle(im, (startX,startY), (endX,endY),color, thick)
    # display output
    # noinspection PyUnresolvedReferences
    img_rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    # org
    #co-ordinates of text
    org = (100,90)
    # Using cv2.putText() method
    # noinspection PyUnresolvedReferences
    image = cv2.putText(img_rgb, user_input, org, font,
                        fontScale, color_text, thickness, cv2.LINE_AA)

    st.image(img_rgb, caption='Voila Face Detected...')
