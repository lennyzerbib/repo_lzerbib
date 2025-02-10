import streamlit as st
from PIL import Image
from streamlit_pdf_viewer import pdf_viewer
import os

import sys
sys.path.append('/Users/lennyzerbib/Desktop/Dauphine/streamlit_page/needs')
import script_cover_letter as script

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        min-width: 200px;
        max-width: 250px;
        }
    """,
    unsafe_allow_html=True,
    )

container_pdf, container_chat = st.columns([50, 50])

st.title("Generate your Cover Letter:")

col_1, col_2, col_3 = st.columns(3)
name_boss = col_1.text_input('Enter the Name of the Manager:')
starting_date = col_2.text_input('Enter the Starting Date:')
bank_user = col_3.text_input('Enter the Bank:')

col_1, col_2, col_3 = st.columns(3)
role_user = col_1.text_input('Enter the Role:')
position_user = col_2.text_input('Enter the Position:')
perso_adress = col_3.selectbox('Enter the Area:', ['','Paris', 'London', 'New York'])

outputdir = '/Users/lennyzerbib/Desktop/Dauphine/streamlit_page/Output/'
outputdir_aux = '/Users/lennyzerbib/Desktop/Dauphine/streamlit_page/Output/'

st.write("")
st.write('---')
st.write("")

if len(name_boss)!=0 and len(starting_date)!=0 and len(perso_adress)!=0 and len(role_user)!=0 and len(bank_user)!=0 and len(position_user)!=0:
    cover_currentname = f'cover_letter_{bank_user.replace(" ", "_")}'
    os.makedirs(outputdir_aux,exist_ok=True)
    f = open(f"{outputdir_aux}{cover_currentname}.tex", "w")
    new_string = script.generate_cover_letter(script.generate_adress(perso_adress), position_user, name_boss, role_user, bank_user, starting_date)
    f.write(new_string)
    f.close()
    os.system(f'pdflatex -no-shell-escape -aux-directory={outputdir_aux} -output-directory={outputdir}'+' '+f'{outputdir_aux}{cover_currentname}.tex')
    st.header('Result:')
    pdf_viewer(input=f'{outputdir}{cover_currentname}.pdf', width=700,height=500)
    #TODO cremove all files within folder
    for file in os.listdir(outputdir_aux):
        if file.endswith('.log') or file.endswith('.out') or file.endswith('.aux') or file.endswith('.tex'):   
            os.remove(outputdir_aux+file)
    
else:
    st.header('Exemple:')
    pdf_viewer(input=r'/Users/lennyzerbib/Desktop/Dauphine/streamlit_page/needs/exemple_cover_letter.pdf', width=700,height=500)