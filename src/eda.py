import streamlit as st
import pandas as pd

# Reuse this data across runs!
read_and_cache_csv = st.cache(pd.read_csv)

# BUCKET = 'data/raw/breast-cancer-wisconsin.data'
# data = read_and_cache_csv(BUCKET + "labels.csv.gz", nrows=100)
desired_label = st.selectbox('Filter to:', ['car', 'truck'])
st.write(data[data.label == desired_label])

class Explorer:
    '''
    Class to explore the dataset
    '''
    