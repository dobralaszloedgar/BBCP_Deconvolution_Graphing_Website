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
</style>
""", unsafe_allow_html=True)


# Function to create a card
def create_card(title, description, icon, app_name):
    # Encode the app name for the URL
    encoded_app = base64.b64encode(app_name.encode()).decode()

    # Create the card with HTML
    card_html = f"""
    <div class="card" onclick="window.location.href='?app={encoded_app}';">
        <div class="card-content">
            <div class="icon">{icon}</div>
            <div class="title">{title}</div>
            <div class="description">{description}</div>
        </div>
    </div>
    """
    return card_html


# Main app
def main():
    # Check if an app has been selected first
    query_params = st.query_params
    if "app" in query_params:
        try:
            # Get the app parameter value
            app_param = query_params["app"]

            # Decode the base64 encoded app name
            selected_app = base64.b64decode(app_param).decode()

            if selected_app == "gaussian_deconvolution":
                # Clear query parameters to prevent reloading issues
                st.query_params.clear()

                # Import and run the Gaussian Deconvolution app
                from gaussian_deconvolution import main as gaussian_main
                gaussian_main()
                return
            elif selected_app == "gpc_graphing":
                # Clear query parameters to prevent reloading issues
                st.query_params.clear()

                # Import and run the GPC Graphing app
                from gpc_graphing import main as gpc_main
                gpc_main()
                return

        except Exception as e:
            st.error(f"Error loading application: {str(e)}")
            st.info("Please ensure the application files are available.")

    # If no app selected or error occurred, show the launcher
    st.markdown("<h1 style='text-align: center; margin-bottom: 40px;'>Choose an Application</h1>",
                unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns(2)

    # Gaussian Deconvolution card
    with col1:
        card1 = create_card(
            "Gaussian Deconvolution",
            "Deconvolute chromatogram data into Gaussian peaks for molecular weight analysis",
            "📊",
            "gaussian_deconvolution"
        )
        st.markdown(card1, unsafe_allow_html=True)

    # GPC Graphing card
    with col2:
        card2 = create_card(
            "GPC Graphing",
            "Create and customize GPC chromatograms with various visualization options",
            "📈",
            "gpc_graphing"
        )
        st.markdown(card2, unsafe_allow_html=True)


if __name__ == "__main__":
    main()