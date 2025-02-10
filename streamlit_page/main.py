import sys
import streamlit.web.cli as stcli

if __name__ == '__main__':
    sys.argv = ["streamlit","run","~/Desktop/repo_lzerbib/streamlit_page/Home.py"]
    sys.exit(stcli.main())