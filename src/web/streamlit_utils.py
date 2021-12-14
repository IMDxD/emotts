import streamlit as st


def hide_hamburger_menu() -> None:
    """Hides Streamlit Hamburger menu from user."""
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def st_empty_block(size: int) -> None:
    for _ in range(size):
        st.text("")


def st_header_centered(text: str) -> None:
    st.markdown(f"<h1 style='text-align: center;'>{text}</h1>", unsafe_allow_html=True)
