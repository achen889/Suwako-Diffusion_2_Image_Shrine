import webview

from MewUtilities.mew_core import init_thread, log_info, get_caller_info

from MewUtilities.mew_ui import MewLayout, ui, mew_ui_notify

from MewUtilities.mew_streamlit import *


import webview

port = 7777


def main_webview(app_name, st_script_file, port):

    streamlit_thread = init_thread(run_st, st_script_file, port)

    if streamlit_thread.is_alive():
        webview.create_window(f"{app_name}", url=f"http://localhost:{port}",  width=1600, height=900, resizable=True)

    #webview.evaluate_js('document.body.innerHTML = "<h1>main_webview</h1>";')

    # Add a button
    #webview.create_button("Click me", on_button_click)

    webview.start(debug=True)


def log_text_input(key : str):
    log_info(f'===[{key}]: ')
    input_text = input()
    #log_info(f'got input_text: {input_text}')
    return input_text


st_script_file = log_text_input("script_file")
app_name = st_script_file + " web view"#log_text_input("app_name")

if __name__ == "__main__":
    main_webview(app_name.upper(), st_script_file, port)

