import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Gaussian Launcher",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling with dark mode support
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
    background-color: var(--background-color);
    margin: 10px;
    border: 1px solid var(--border-color);
}

/* Light mode colors */
[data-theme="light"] .card {
    --background-color: white;
    --border-color: #e0e0e0;
    --text-color: #333;
}

/* Dark mode colors */
[data-theme="dark"] .card {
    --background-color: #262730;
    --border-color: #454545;
    --text-color: #fafafa;
}

/* Auto-detect system theme */
@media (prefers-color-scheme: dark) {
    .card {
        --background-color: #262730;
        --border-color: #454545;
        --text-color: #fafafa;
    }
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
    color: var(--text-color);
}

.description {
    font-size: 16px;
    color: var(--text-color);
    opacity: 0.8;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for navigation
if 'current_app' not in st.session_state:
    st.session_state.current_app = None


# Function to create a card
def create_card(title, description, icon, app_name):
    # Create the card with HTML
    card_html = f"""
    <div class="card" onclick="document.getElementById('{app_name}').click()">
        <div class="card-content">
            <div class="icon">{icon}</div>
            <div class="title">{title}</div>
            <div class="description">{description}</div>
        </div>
    </div>
    <button id="{app_name}" style="display:none;"></button>
    """
    return card_html


# Check if we should show a specific app
if st.session_state.current_app == "deconvolution":
    # Import and run the deconvolution app
    exec(open('gaussian_deconvolution.py').read())
else:
    # Show the launcher interface
    st.title("🚀 Gaussian Analysis Suite")
    st.markdown("Choose an application to get started:")

    # Create columns for cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(create_card(
            "Gaussian Deconvolution",
            "Analyze and deconvolve Gaussian peaks from your chromatography data",
            "📊",
            "deconvolution"
        ), unsafe_allow_html=True)

        # Button for navigation (hidden but clickable)
        if st.button("Launch Deconvolution", key="deconvolution", help="Click to open Gaussian Deconvolution"):
            st.session_state.current_app = "deconvolution"
            st.rerun()

    with col2:
        st.markdown(create_card(
            "Data Visualization",
            "Create custom plots and visualizations for your data",
            "📈",
            "visualization"
        ), unsafe_allow_html=True)

        if st.button("Coming Soon", key="visualization", disabled=True):
            pass

    with col3:
        st.markdown(create_card(
            "Statistical Analysis",
            "Perform statistical analysis on your datasets",
            "📉",
            "statistics"
        ), unsafe_allow_html=True)

        if st.button("Coming Soon", key="statistics", disabled=True):
            pass

    # Add some spacing and info
    st.markdown("---")
    st.markdown("**Note:** Click on any card to launch the corresponding application.")
