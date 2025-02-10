import streamlit as st
import sys
sys.path.append(r'~/Desktop/repo_lzerbib/streamlit_page')
import datetime as dt
import time

st.set_page_config(layout='wide')

st.title('Welcome on Lenny\'s Page')

st.markdown('#This is where magic happens')

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

st.write('In case data have not shown, press R')

placeholder = st.empty()

while True:
    now_paris = dt.datetime.now()
    now_london = now_paris - dt.timedelta(hours=1)
    now_hk = now_paris + dt.timedelta(hours=6)
    now_ny = now_paris - dt.timedelta(hours=6)
    
    with placeholder.container():
        
        paris_col, london_col, hk_col, ny_col = st.columns(4)
        
        paris_col.markdown('## Paris')
        london_col.markdown('## London')
        hk_col.markdown('## Hong Kong')
        ny_col.markdown('## New York')
        
        paris_col.markdown(f'### {now_paris.strftime("%H:%M")}')
        london_col.markdown(f'### {now_london.strftime("%H:%M")}')
        hk_col.markdown(f'### {now_hk.strftime("%H:%M")}')
        ny_col.markdown(f'### {now_ny.strftime("%H:%M")}')
        
        time.sleep(1)