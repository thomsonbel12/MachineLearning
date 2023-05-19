import streamlit as st
import numpy as np
import cv2 as cv
import joblib
st.set_page_config(page_title="Face Detection", page_icon="🔍")
st.subheader('Face Detection')
FRAME_WINDOW = st.image([])
cap = cv.VideoCapture(0)

if 'stop' not in st.session_state:
    st.session_state.stop = False
    stop = False

press = st.button('Stop')
if press:
    if st.session_state.stop == False:
        st.session_state.stop = True
        cap.release()
    else:
        st.session_state.stop = False

print('Trang thai nhan Stop', st.session_state.stop)

if 'frame_stop' not in st.session_state:
    frame_stop = cv.imread('./pages/stop.jpg')
    st.session_state.frame_stop = frame_stop
    print('Đã load stop.jpg')

if st.session_state.stop == True:
    FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')

#Tạo một biến svc bằng cách tải mô hình đã được huấn luyện trước từ tệp svc.pkl sử dụng joblib.load.
svc = joblib.load('./pages/svc.pkl')
# Định nghĩa một danh sách mydict chứa tên của các người trong mô hình nhận dạng đã được train.
mydict = ['BanKiet', 'BanNghia',  'BanThanh','HoanHao', 'HoangLam', 'HuyTruong', 'SangSang', 'ThayDuc']

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            #print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


if __name__ == '__main__':
    # Đọc mô hình nhận diện khuôn mặt từ tệp face_detection_yunet_2022mar.onnx
    detector = cv.FaceDetectorYN.create(
        './pages/face_detection_yunet_2022mar.onnx',
        "",
        (320, 320),
        0.9,
        0.3,
        5000)
    #Đọc mô hình nhận dạng khuôn mặt từ tệp face_recognition_sface_2021dec.onnx và tạo một đối tượng recognizer
    recognizer = cv.FaceRecognizerSF.create(
    './pages/face_recognition_sface_2021dec.onnx',"")

    
    tm = cv.TickMeter()

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    dem = 0
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Inference
        tm.start()
        # Sử dụng detector.detect để nhận diện các khuôn mặt trong khung hình. Kết quả trả về là một tuple faces.
        faces = detector.detect(frame) # faces is a tuple
        tm.stop()
        
        # Nếu có ít nhất một khuôn mặt được nhận diện (faces[1] is not None), thực hiện các bước nhận dạng khuôn mặt.
        if faces[1] is not None:
            face_align = recognizer.alignCrop(frame, faces[1][0])
            face_feature = recognizer.feature(face_align)
            test_predict = svc.predict(face_feature)
            result = mydict[test_predict[0]]
            # Vẽ tên của người được dự đoán lên khung hình bằng cv.putText.
            cv.putText(frame,result,(1,50),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw results on the input image
        # Gọi hàm visualize để vẽ các khuôn mặt đã được nhận diện lên khung hình.
        visualize(frame, faces, tm.getFPS())

        # Visualize results
        # Hiển thị khung hình đầu vào với các khuôn mặt đã được nhận diện bằng FRAME_WINDOW.image.
        FRAME_WINDOW.image(frame, channels='BGR')
    cv.destroyAllWindows()
