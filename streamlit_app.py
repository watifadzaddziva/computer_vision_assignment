import streamlit as st
import cv2
import tensorflow as tf 
import numpy as np
from keras.models import load_model

#Loading the Inception model
model= load_model('model.h5',compile=(False))



#Functions
def predict(frame, model):
    # Pre-process the image for model prediction
    img = cv2.resize(frame, (299, 299))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img /= 255.0

    # Predict with the Inceptionv3 model
    prediction = model.predict(img)

    # Convert the prediction into text
    pred_text = tf.keras.applications.inception_v3.decode_predictions(prediction, top=1)
    for (i, (imagenetID, label, prob)) in enumerate(pred_text[0]):
        label  = ("{}: {:.2f}%".format(label, prob * 100))

    st.markdown(label)


def object_detection(frame, model):
    # Pre-process the image for model prediction
    img = cv2.resize(frame, (299, 299))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img /= 255.0

    # Predict with the Inceptionv3 model
    prediction = model.predict(img)

    # Convert the prediction into text
    pred_text = tf.keras.applications.inception_v3.decode_predictions(prediction, top=1)
    for (i, (imagenetID, label, prob)) in enumerate(pred_text[0]):
        # Get the name of the predicted class
        pred_class = label

        # Get the probability of the prediction
        pred_probab = prob

    # Get the x and y coordinates of the top left corner of the predicted class box
    pred_x1 = frame.shape[0] * pred_text[3][0]
    pred_y1 = frame.shape[1] * pred_text[3][1]

    # Get the x and y coordinates of the bottom right corner of the predicted class box
    pred_x2 = frame.shape[0] * pred_text[3][2]
    pred_y2 = frame.shape[1] * pred_text[3][3]

    # Display the prediction
    cv2.rectangle(frame, (int(pred_x1), int(pred_y1)), (int(pred_x2), int(pred_y2)), (0, 255, 0), 4)
    cv2.putText(frame, pred_class + " " + str(pred_probab), (int(pred_x1), int(pred_y1)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame

# Main App
def main():
    """Deployment using Streamlit"""
    st.title("Object Detection App")
    st.text("Built with Streamlit and Inceptionv3")

    activities = ["Detect Objects", "About"]
    choice = st.sidebar.selectbox("Choose Activity", activities)
    
    if choice == "Detect Objects":
        st.subheader("Upload Video")

        video_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])

        if video_file is not None:
            path = video_file.name
            with open(path,mode='wb') as f: 
                f.write(video_file.read())         
                st.success("Saved File")
            cap = cv2.VideoCapture(path)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))
            
            if st.button("Detect Objects"):
                
                # Start the video prediction loop
                while cap.isOpened():
                    ret, frame = cap.read()
    
                    if not ret:
                        break
    
                    # Perform object detection
                    #frame = object_detection(frame, model)
                    frame = predict(frame, model)
    
                    # Display the resulting frame
                    #st.image(frame, caption='Video Stream', use_column_width=True)
    
                cap.release()
                output.release()
                cv2.destroyAllWindows()
                
            elif st.button("Search for an object"):
                
                key = st.text_input('Search key')

                # Start the video prediction loop
                while cap.isOpened():
                    ret, frame = cap.read()
    
                    if not ret:
                        break
    
                    # Perform object detection
                    frame = object_detection(frame, model)
                    #frame = predict(frame, model)
    
                    # Display the resulting frame
                    #st.image(frame, caption='Video Stream', use_column_width=True)
    
                cap.release()
                output.release()
                cv2.destroyAllWindows()

    elif choice == "About":
        st.subheader("About")
        st.text("Built with Streamlit and Inceptionv3")
        st.subheader("By")
        st.text("Watifadza Dziva\nThabolezwe Mabandla")


if __name__ == '__main__':
    main()
