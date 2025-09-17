import streamlit as st
import base64

# Page config
st.set_page_config(
    page_title="Gaussian Launcher",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Theming + card styles (dark-mode aware, using Streamlit theme vars with sensible fallbacks)
st.markdown("""
<style>
:root {
  --bg: var(--background-color, #0e1117);
  --fg: var(--text-color, #e5e7eb);
  --secondary-bg: var(--secondary-background-color, #1f2937);
  --muted: #9ca3af;
  --muted-light: #6b7280;
  --card-bg-light: #ffffff;
  --card-bg-dark: #1f2937;
  --border-light: rgba(0,0,0,0.08);
  --border-dark: rgba(255,255,255,0.08);
}
.stApp { background-color: var(--bg); color: var(--fg); }
h1 { color: var(--fg); }

/* Anchor wrapper so the whole card is clickable */
.card-link {
  text-decoration: none;
  color: inherit;
  width: 100%;
  display: block;
  position: relative;
}

/* Card base */
.card {
  padding: 20px;
  border-radius: 14px;
  box-shadow: 0 8px 20px rgba(0,0,0,0.25);
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
  height: 300px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  border: 1px solid var(--border-dark);
  background-color: var(--secondary-bg);
  color: var(--fg);
}

/* Light mode overrides */
@media (prefers-color-scheme: light) {
  .card {
    background-color: var(--card-bg-light);
    color: #111827;
    border: 1px solid var(--border-light);
  }
  .description { color: var(--muted-light) !important; }
}

/* Dark mode overrides */
@media (prefers-color-scheme: dark) {
  .card {
    background-color: var(--card-bg-dark);
    color: #e5e7eb;
    border: 1px solid var(--border-dark);
  }
  .description { color: var(--muted) !important; }
}

/* Hover state for enabled cards */
.card:hover {
  box-shadow: 0 12px 28px rgba(0,0,0,0.35);
  transform: translateY(-4px);
}

/* Inner layout */
.card-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
}
.icon { font-size: 60px; margin-bottom: 20px; }
.title { font-size: 24px; font-weight: bold; margin-bottom: 10px; }
.description { font-size: 16px; }

/* Disabled state (Coming soon) */
.card.disabled {
  opacity: 0.45;
  filter: grayscale(20%);
  pointer-events: none;
  cursor: not-allowed;
  transform: none;
  box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}

/* Badge */
.badge {
  position: absolute;
  top: 12px;
  right: 12px;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,.15);
  background: rgba(107,114,128,.18);
  color: inherit;
}

/* Ensure the section header stays readable */
.block-container h1 { color: inherit; }

</style>
""", unsafe_allow_html=True)


def create_card(title: str, description: str, icon: str, app_name: str, *, disabled: bool = False, badge: str | None = None) -> str:
    classes = "card disabled" if disabled else "card"
    if disabled:
        # Not clickable; show a non-link wrapper
        return f"""
        <div class="card-link">
          <div class="{classes}">
            {'<div class="badge">'+badge+'</div>' if badge else ""}
            <div class="card-content">
                <div class="icon">{icon}</div>
                <div class="title">{title}</div>
                <div class="description">{description}</div>
            </div>
          </div>
        </div>
        """
    else:
        # Clickable via query param navigation
        return f"""
        <a class="card-link" href="?app={app_name}" target="_self" aria-label="Open {title}">
          <div class="{classes}">
            {'<div class="badge">'+badge+'</div>' if badge else ""}
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
        qp = st.query_params
        app_val = qp.get("app")
    except Exception:
        try:
            qp = st.experimental_get_query_params()
            app_val = qp.get("app")
        except Exception:
            app_val = None
    if isinstance(app_val, list):
        return app_val if app_val else None
    return app_val


def _back_to_launcher_button():
    if st.button("← Back to Launcher"):
        try:
            st.query_params.clear()
        except Exception:
            try:
                st.experimental_set_query_params()
            except Exception:
                pass
        st.rerun()


def main():
    selected_app = _get_selected_app()

    if selected_app == "gaussian_deconvolution":
        from gaussian_deconvolution import main as gaussian_main
        gaussian_main()
        return

    # Hard-block navigation to GPC Graphing while it's under development
    if selected_app == "gpc_graphing":
        st.info("GPC Graphing is under development and will be available soon.", icon="🛠️")
        _back_to_launcher_button()
        return

    # Launcher UI
    st.markdown("<h1 style='text-align: center; margin-bottom: 40px;'>Choose an Application</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(create_card(
            "Gaussian Deconvolution",
            "Deconvolute chromatogram data into Gaussian peaks for molecular weight analysis",
            "📊",
            "gaussian_deconvolution",
            disabled=False
        ), unsafe_allow_html=True)

    with col2:
        st.markdown(create_card(
            "GPC Graphing",
            "Create and customize GPC chromatograms with various visualization options",
            "📈",
            "gpc_graphing",
            disabled=True,
            badge="Coming soon"
        ), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
