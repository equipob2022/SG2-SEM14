import streamlit as st
from multiapp import MultiApp
from apps import home,modelTwitter # import your app modules here model2

app = MultiApp()

st.markdown("""
#  Inteligencia de Negocios - Grupo B

""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Modelo twitter", modelTwitter.app)
# The main app
app.run()




