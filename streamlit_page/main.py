import sys
import streamlit.web.cli as stcli

if __name__ == '__main__':
    sys.argv = ["streamlit","run","/Users/lennyzerbib/Desktop/Dauphine/streamlit_page/Home.py"]
    sys.exit(stcli.main())