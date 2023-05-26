import streamlit as st 

st.set_page_config(page_title="My Website", page_icon=":middle_finger", layout="wide")

# ---- HEAD ----

with st.container():
    st.header("Test Header")
    st.title("Title")
    st.write("##")
    st.write("[Click This Link >](https://www.youtube.com/watch?v=VqgUkExPvLY&t=570s)")

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('CarBikeDataset.hdf5')
  return model
model=load_model()
st.write("""
# Car-Bike Detection System"""
)
