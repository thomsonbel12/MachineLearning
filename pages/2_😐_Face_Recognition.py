import streamlit as st
import numpy as np
import cv2 as cv
st.set_page_config(page_title="Face Reconition", page_icon="😐")
st.subheader('Face Reconition')
# Tạo Một hình ảnh placeholder FRAME_WINDOW được tạo bằng st.image([]).
FRAME_WINDOW = st.image([])
# Thiết lập các biến deviceId và cap để truy cập camera. Trong trường hợp này, deviceId được đặt là 0, tương ứng với camera mặc định.
deviceId = 0
cap = cv.VideoCapture(deviceId)

# Kiểm tra nếu 'stop' không có trong st.session_state, sau đó tạo biến stop và st.session_state.stop với giá trị mặc định False.
if 'stop' not in st.session_state:
    st.session_state.stop = False
    stop = False


press = st.button('Stop')
# Khi nút 'Stop' được nhấn
if press:
    #  Nếu st.session_state.stop là False, thì camera sẽ được dừng và st.session_state.stop được thiết lập là True.
    if st.session_state.stop == False:
        st.session_state.stop = True
        cap.release()
    # Ngược lại
    else:
        st.session_state.stop = False

print('Trang thai nhan Stop', st.session_state.stop)

if 'frame_stop' not in st.session_state:
    frame_stop = cv.imread('./pages/stop.jpg')
    st.session_state.frame_stop = frame_stop
    print('Đã load stop.jpg')

if st.session_state.stop == True:
    FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')

# Định nghĩa hàm visualize để hiển thị các khuôn mặt được nhận diện lên hình ảnh đầu vào.
def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            # print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Tạo một đối tượng detector của lớp FaceDetectorYN từ OpenCV, sử dụng mô hình nhận diện khuôn mặt đã được đào tạo trước đó.
detector = cv.FaceDetectorYN.create(
    './pages/face_detection_yunet_2022mar.onnx',
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)
# Thiết lập kích thước khung hình đầu vào cho bộ nhận diện khuôn mặt.
tm = cv.TickMeter()
frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
detector.setInputSize([frameWidth, frameHeight])

# Trong vòng lặp vô hạn, đọc từng khung hình từ camera sử dụng cap.read().
while True:
    hasFrame, frame = cap.read()
    # Kiểm tra nếu không có khung hình, tức là không thể đọc được từ camera, thì vòng lặp dừng lại.
    if not hasFrame:
        print('No frames grabbed!')
        break

    frame = cv.resize(frame, (frameWidth, frameHeight))

    # Inference
    tm.start()
    # Thực hiện nhận diện khuôn mặt bằng cách gọi phương thức detect của detector.
    faces = detector.detect(frame) # faces is a tuple
    tm.stop()

    # Draw results on the input image
    # Gọi hàm visualize để vẽ các khuôn mặt được nhận diện lên khung hình đầu vào.
    visualize(frame, faces, tm.getFPS())

    # Visualize results
    # Hiển thị khung hình đầu vào với các khuôn mặt đã được nhận diện bằng FRAME_WINDOW.image.
    FRAME_WINDOW.image(frame, channels='BGR')
cv.destroyAllWindows()
