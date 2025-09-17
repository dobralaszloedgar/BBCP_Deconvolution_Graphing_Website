import streamlit as st
import base64

# Set page configuration
st.set_page_config(
    page_title="Gaussian Launcher",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    .card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        height: 300px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        cursor: pointer;
        background-color: white;
    }
    .card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
        transform: translateY(-5px);
    }
    .card-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
    }
    .icon {
        font-size: 60px;
        margin-bottom: 20px;
    }
    .title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .description {
        font-size: 16px;
        color: #666;
    }
    a.card-link {
        text-decoration: none;
        color: inherit;
        width: 100%;
        display: block;
    }
</style>
""", unsafe_allow_html=True)


def create_card(title: str, description: str, icon: str, app_name: str) -> str:
    # Wrap the card in an anchor link instead of inline JS
    return f"""
    <a class="card-link" href="?app={app_name}" target="_self">
      <div class="card">
        <div class="card-content">
            <div class="icon">{icon}</div>
            <div class="title">{title}</div>
            <div class="description">{description}</div>
        </div>
      </div>
    </a>
    """


def _get_selected_app():
    # Handle Streamlit versions with either st.query_params or experimental_get_query_params
    app_val = None
    try:
        qp = st.query_params  # New API (dict-like)
        app_val = qp.get("app")
    except Exception:
        try:
            qp = st.experimental_get_query_params()  # Old API returns dict of lists
            app_val = qp.get("app")
        except Exception:
            app_val = None
    # Normalize value
    if isinstance(app_val, list):
        return app_val[0] if app_val else None
    return app_val


def main():
    # Router
    selected_app = _get_selected_app()
    if selected_app == "gaussian_deconvolution":
        from gaussian_deconvolution import main as gaussian_main
        gaussian_main()
        return
    elif selected_app == "gpc_graphing":
        try:
            from gpc_graphing import main as gpc_main
            gpc_main()
            return
        except Exception as e:
            st.error(f"GPC Graphing could not be loaded: {e}")

    # Launcher UI
    st.markdown("<h1 style='text-align: center; margin-bottom: 40px;'>Choose an Application</h1>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(create_card(
            "Gaussian Deconvolution",
            "Deconvolute chromatogram data into Gaussian peaks for molecular weight analysis",
            "📊",
            "gaussian_deconvolution"
        ), unsafe_allow_html=True)

    with col2:
        st.markdown(create_card(
            "GPC Graphing",
            "Create and customize GPC chromatograms with various visualization options",
            "📈",
            "gpc_graphing"
        ), unsafe_allow_html=True)

    # Optional: Pure-Streamlit fallback navigation (no query params) if preferred:
    # with col1:
    #     if st.button("Open Gaussian Deconvolution", use_container_width=True):
    #         from gaussian_deconvolution import main as gaussian_main
    #         gaussian_main()
    #         st.stop()


if __name__ == "__main__":
    main()
