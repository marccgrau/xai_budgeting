import streamlit as st
from streamlit_option_menu import option_menu
from pages.intro_page import intro_page
from pages.data_analysis_page import data_analysis_page
from pages.forecasting_page import forecasting_page
from pages.results_page import results_page

st.set_page_config(page_title="Budget Forecasting App", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none}
    section[data-testid="stSidebar"] {display: none !important;}
    .stApp {
        max-width: 100%;
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["Intro", "Data Analysis", "Forecasting", "Results"],
    icons=["house", "bar-chart-line", "graph-up-arrow"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {
            "font-size": "25px",
            "text-align": "left",
            "margin":"0px",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {"background-color": "lightblue"},
    }
)

if selected == "Intro":
    intro_page()
elif selected == "Data Analysis":
    data_analysis_page()
elif selected == "Forecasting":
    forecasting_page()
elif selected == "Results":
    results_page()

