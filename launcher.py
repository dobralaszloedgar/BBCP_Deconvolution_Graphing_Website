import streamlit as st
import base64

# Page config
st.set_page_config(
    page_title="Gaussian Launcher",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark-mode aware styling with subtle shadows and no washed-out glow
st.markdown("""
<style>
:root {
  --bg: var(--background-color, #0e1117);
  --fg: var(--text-color, #e5e7eb);
  --surface-dark: rgba(255,255,255,0.04);
  --surface-light: #ffffff;
  --border-dark: rgba(255,255,255,0.10);
  --border-light: rgba(0,0,0,0.08);
  --muted-dark: #a3a3a3;
  --muted-light: #6b7280;
}
.stApp { background-color: var(--bg); color: var(--fg); }
h1 { color: var(--fg); }

.card-grid { margin-top: 8px; }

/* Make the link wrapper neutral */
.card-link {
  text-decoration: none;
  color: inherit;
  display: block;
  width: 100%;
  background: transparent;
}

/* Reduce default focus ring on the anchor, but keep accessibility */
.card-link:focus-visible { outline: 2px solid #3b82f6; outline-offset: 4px; border-radius: 16px; }

/* Card base */
.card {
  position: relative;
  padding: 22px;
  border-radius: 16px;
  height: 300px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;

  /* Dark defaults */
  background: var(--surface-dark);
  border: 1px solid var(--border-dark);
  color: var(--fg);

  /* Subtle shadow only (remove big bloom) */
  box-shadow: 0 4px 14px rgba(0,0,0,0.20);
  transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease, background-color 120ms ease;
}

/* Light mode overrides */
@media (prefers-color-scheme: light) {
  .card {
    background: var(--surface-light);
    border: 1px solid var(--border-light);
    color: #111827;
    box-shadow: 0 3px 10px rgba(0,0,0,0.10);
  }
}

/* Hover state: very subtle lift, no glowing halo */
.card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 18px rgba(0,0,0,0.22);
}

/* Content */
.card-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
}

.icon { font-size: 56px; margin-bottom: 16px; }
.title { font-size: 22px; font-weight: 700; margin-bottom: 8px; }
.description { font-size: 15px; color: var(--muted-dark); line-height: 1.45; }

/* Light-mode muted text */
@media (prefers-color-scheme: light) {
  .description { color: var(--muted-light); }
}

/* Disabled state (Coming soon) */
.card.disabled {
  opacity: 0.5;
  filter: grayscale(15%);
  pointer-events: none;
  cursor: not-allowed;
  transform: none;
  box-shadow: 0 3px 10px rgba(0,0,0,0.14);
}

/* Badge */
.badge {
  position: absolute;
  top: 10px;
  right: 10px;
  font-size: 11px;
  padding: 3px 8px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(107,114,128,0.22);
  color: inherit;
}

/* Reduce any container bleed that can look like a white panel edge */
.block-container { padding-top: 1.5rem; }

/* Keep header centered */
.block-container h1 { text-align: center; margin-bottom: 40px; }
</style>
""", unsafe_allow_html=True)


def create_card(title: str, description: str, icon: str, app_name: str, *, disabled: bool = False, badge: str | None = None) -> str:
    classes = "card disabled" if disabled else "card"
    badge_html = f'<div class="badge">{badge}</div>' if badge else ""
    if disabled:
        # Non-clickable
        return f"""
        <div class="card-link">
          <div class="{classes}">
            {badge_html}
            <div class="card-content">
                <div class="icon">{icon}</div>
                <div class="title">{title}</div>
                <div class="description">{description}</div>
            </div>
          </div>
        </div>
        """
    # Clickable
    return f"""
    <a class="card-link" href="?app={app_name}" target="_self" aria-label="Open {title}">
      <div class="{classes}">
        {badge_html}
        <div class="card-content">
            <div class="icon">{icon}</div>
            <div class="title">{title}</div>
            <div class="description">{description}</div>
        </div>
      </div>
    </a>
    """


def _get_selected_app():
    # Works across Streamlit versions
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

    # Block GPC Graphing route if someone hand-types the URL
    if selected_app == "gpc_graphing":
        st.info("GPC Graphing is under development and will be available soon.", icon="🛠️")
        _back_to_launcher_button()
        return

    # Launcher UI
    st.markdown("### Choose an Application")

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
