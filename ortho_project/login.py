import streamlit as st
from PIL import Image
import app
import hashlib
import sqlite3
import os

# Clear all Streamlit cache
st.cache_data.clear()
st.cache_resource.clear()

def get_custom_css(bg_color):
    return f"""
    <style>
        .stApp {{
            background-color: {bg_color} !important;
        }}
        .stTextInput>div>div>input::placeholder {{
            color: transparent;
        }}
        .stTextInput label {{
            color: {'white' if bg_color == '#201E45' else 'black'} !important;
        }}
    </style>
    """

# At the start of your script, set the initial background color
if 'bg_color' not in st.session_state:
    st.session_state.bg_color = '#201E45'  # dark background for login page

# Apply the custom CSS
st.markdown(get_custom_css(st.session_state.bg_color), unsafe_allow_html=True)

# Load the logo image
logo_image = Image.open("Pro-Mapp Health.png")  # Replace with your logo file name

# DB Management functions
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

conn = sqlite3.connect('user_data.db')
c = conn.cursor()

def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_userdata(username, password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)', (username, password))
    conn.commit()

def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, password))
    data = c.fetchall()
    return data

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ''
    st.session_state.bg_color = '#201E45'  # Revert to dark background
    st.rerun()

def show_login():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if 'username' not in st.session_state:
        st.session_state.username = ''

    # Main page content
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        st.image(logo_image, use_column_width=True)
    
    # Sidebar content
    st.sidebar.title("Login / Sign Up")
    
    menu = ["Login", "Sign Up"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Login":
        st.sidebar.subheader("Login Section")
        
        username = st.sidebar.text_input("User Name", key="login_username")
        password = st.sidebar.text_input("Password", type='password', key="login_password")
        if st.sidebar.button("Login", key="login_button"):
            create_usertable()
            hashed_pswd = make_hashes(password)
            result = login_user(username, check_hashes(password, hashed_pswd))
            if result:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.bg_color = 'white'  # Change background to white
                st.success("Logged In Successfully")
                st.rerun()
            else:
                st.sidebar.warning("Incorrect Username/Password")
    
    elif choice == "Sign Up":
        st.sidebar.subheader("Create New Account")
        new_user = st.sidebar.text_input("Username", key="signup_username")
        new_password = st.sidebar.text_input("Password", type='password', key="signup_password")
        
        if st.sidebar.button("Sign Up", key="signup_button"):
            create_usertable()
            add_userdata(new_user, make_hashes(new_password))
            st.sidebar.success("You have successfully created a valid Account")
            st.sidebar.info("Go to Login Menu to login")

    return st.session_state.logged_in

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if 'bg_color' not in st.session_state:
        st.session_state.bg_color = '#201E45'  # Initialize with dark background

    # Apply the custom CSS
    st.markdown(get_custom_css(st.session_state.bg_color), unsafe_allow_html=True)

    if not st.session_state.logged_in:
        show_login()
    else:
        # Add a logout button
        if st.sidebar.button("Logout", key="logout_button"):
            logout()
        else:
            st.session_state.bg_color = 'white'  # Set background to white when logged in
            app.main_app_medacta()

if __name__ == '__main__':
    main()