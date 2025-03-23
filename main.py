import streamlit as st
import pandas as pd
import hashlib
import os

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# File to store user credentials
USER_DATA_FILE = "users.csv"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def initialize_csv():
    if not os.path.exists(USER_DATA_FILE):
        df = pd.DataFrame(columns=["username", "password"])
        df.to_csv(USER_DATA_FILE, index=False)

def user_exists(username):
    df = pd.read_csv(USER_DATA_FILE)
    return username in df["username"].values

#add a new user
def add_user(username, password):
    df = pd.read_csv(USER_DATA_FILE)
    hashed_password = hash_password(password)
    new_user = pd.DataFrame([[username, hashed_password]], columns=["username", "password"])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USER_DATA_FILE, index=False)

#user authentication
def authenticate(username, password):
    df = pd.read_csv(USER_DATA_FILE)
    hashed_password = hash_password(password)
    user_row = df[df["username"] == username]
    return not user_row.empty and user_row.iloc[0]["password"] == hashed_password

initialize_csv()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    st.switch_page("pages/app.py")

st.title("Login")

#Sign in
def signin():
    st.markdown("### Sign In to your account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    col1, col2 = st.columns([87, 13])
    with col1:
        if st.button("Login"):
            if authenticate(username, password):
                st.success(f"Welcome, {username}!")
                st.session_state.logged_in = True

                st.switch_page("pages/app.py")  
            else:
                st.error("Invalid username or password.")
    
    with col2:
        if st.button("Sign Up"):
            st.session_state.page = "Sign Up"
            st.rerun()

#Sign up
def signup():
    st.subheader("Create a new account")
    new_user = st.text_input("New Username")
    new_pass = st.text_input("New Password", type="password")
    
    col1, col2 = st.columns([87, 13])
    with col1:
        if st.button("Sign Up"):
            if user_exists(new_user):
                st.warning("Username already exists! Choose another.")
            elif new_user and new_pass:
                add_user(new_user, new_pass)
                st.success("Registration successful! You can now Sign In.")
            else:
                st.error("Please fill all fields.")
    
    with col2:
        if st.button("Sign In"):
            st.session_state.page = "Sign In"
            st.rerun()

if "page" not in st.session_state or st.session_state.page == "Sign In":
    signin()
else:
    signup()

