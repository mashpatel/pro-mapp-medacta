import streamlit as st
from streamlit_aggrid import AgGrid

# Sample data
data = {'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']}

# Create a grid
AgGrid(data)