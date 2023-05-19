import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
     """
    Build a Machine Learning web application in Python with Streamlit. 
    
    ### Contents in the project:
    - Handwriting recognition
    - Face recognition
    - Face detection
    - California house price prediction
    - Languge Detection
    - Finger Count
    ### How to run this application
    - Open project in VSCode
    - Open Terminal in VSCode
    - Run "streamlit run 👋main.py
    - If there is an error please import the library and run it again
    ### Student
    - 20110592 - Phan Công Tú
    - 20110514 - Nguyễn Sỹ Hoàng Lâm
    
    ### Teacher
    - Trần Tiến Đức
    ### Source code
    - code: https://github.com/CunoVox/MachineLearning.git
    ### Reference
    - Teacher Tran Tien Duc
    - https://www.youtube.com/watch?v=1xtrIEwY_zY
    - https://www.datacamp.com/tutorial/streamlit
"""
)
