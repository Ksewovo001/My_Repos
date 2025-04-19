📘 README
We’re excited to introduce the ISU Parents Q&A Chatbot, a new digital assistant designed to support the families of Illinois State University students! Whether you have questions about tuition payments, housing, academic programs, or campus resources, our chatbot is here to provide quick and reliable answers—anytime, anywhere. We invite you to give it a try and explore how this helpful tool can make your ISU journey easier and more informed.

st.markdown("""
    <h1 style='color: #d62828;'>🎓 ISU Parents Q&A Chatbot 🧾</h1>
    <p>Helping parents and families with quick and reliable answers.</p>
""", unsafe_allow_html=True)

cols = st.columns([1, 1, 1])  # Equal width columns for symmetry

try:
    with cols[0]:
        image1 = Image.open("Dr_Birdiclopedia2.png")
        st.image(image1, width=200)
    with cols[2]:
        image2 = Image.open("Dr_Birdiclopedia1.png")
        st.image(image2, width=200)
except Exception as e:
    st.error("One or more images could not be loaded.")
    st.exception(e)
