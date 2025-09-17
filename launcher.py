import streamlit as st

# Page config
st.set_page_config(
    page_title="Gaussian Launcher",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark-mode aware styles and hover "pop" via wrapper hover (works even with overlay link)
st.markdown(
    """
<style>
:root {
  --bg: var(--background-color, #0e1117);
  --fg: var(--text-color, #e5e7eb);
  --surface-dark: rgba(255,255,255,0.04);
  --surface-light: #ffffff;
  --border-dark: rgba(255,255,255,0.12);
  --border-light: rgba(0,0,0,0.08);
  --muted-dark: #a3a3a3;
  --muted-light: #6b7280;
  --accent: #3b82f6;
}
.stApp { background-color: var(--bg); color: var(--fg); }

.header {
  text-align: center;
  margin-bottom: 40px;
  font-size: 2.2rem;
  font-weight: 800;
  color: inherit;
}

/* Wrapper controls hover so overlay link doesn't block :hover on the card */
.card-wrapper {
  position: relative;
  width: 100%;
  transition: transform 140ms ease, filter 140ms ease;
}
.card-wrapper.disabled { pointer-events: none; }

/* Card */
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

  background: var(--surface-dark);
  border: 1px solid var(--border-dark);
  color: var(--fg);

  box-shadow: 0 4px 14px rgba(0,0,0,0.20);
  transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease, background-color 140ms ease;
}

/* Light mode overrides */
@media (prefers-color-scheme: light) {
  .card {
    background: var(--surface-light);
    border: 1px solid var(--border-light);
    color: #111827;
    box-shadow: 0 3px 10px rgba(0,0,0,0.10);
  }
  .description { color: var(--muted-light); }
}

/* POP effect on hover (applies to enabled wrapper) */
.card-wrapper:hover .card {
  transform: translateY(-6px) scale(1.02);
  box-shadow: 0 16px 30px rgba(0,0,0,0.28);
  border-color: color-mix(in srgb, var(--accent) 35%, transparent);
}

/* Prevent disabled from popping */
.card-wrapper.disabled:hover .card {
  transform: none;
  box-shadow: 0 3px 10px rgba(0,0,0,0.14);
  border-color: var(--border-dark);
}

/* Full-size invisible link so the whole card is clickable */
.overlay-link {
  position: absolute;
  inset: 0;
  z-index: 10;
  text-indent: -9999px; /* hide text, keep accessible name */
  background: transparent;
  cursor: pointer;
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

/* Disabled look */
.card.disabled {
  opacity: 0.55;
  filter: grayscale(12%);
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
</style>
""",
    unsafe_allow_html=True,
)

def create_card(title: str, description: str, icon: str, app_name: str, *, disabled: bool = False, badge: str | None = None) -> str:
    badge_html = f'<div class="badge">{badge}</div>' if badge else ""
    wrapper_class = "card-wrapper disabled" if disabled else "card-wrapper"
    card_class = "card disabled" if disabled else "card"
    link_html = "" if disabled else f'<a class="overlay-link" href="?app={app_name}" aria-label="Open {title}">Open {title}</a>'
    # Single-line segments to avoid Markdown code-block interpretation
    return (
        f'<div class="{wrapper_class}">'
        f'  <div class="{card_class}">'
        f'    {badge_html}'
        f'    <div class="card-content">'
        f'      <div class="icon">{icon}</div>'
        f'      <div class="title">{title}</div>'
        f'      <div class="description">{description}</div>'
        f'    </div>'
        f'  </div>'
        f'  {link_html}'
        f'</div>'
    )

def _get_selected_app():
    # Compatible with old/new query param APIs
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

    # Block direct navigation to unfinished app
    if selected_app == "gpc_graphing":
        st.info("GPC Graphing is under development and will be available soon.", icon="🛠️")
        _back_to_launcher_button()
        return

    # Header as HTML (avoid Markdown auto-anchors)
    st.markdown('<div class="header">Choose an Application</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            create_card(
                "Gaussian Deconvolution",
                "Deconvolute chromatogram data into Gaussian peaks for molecular weight analysis",
                "📊",
                "gaussian_deconvolution",
                disabled=False,
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            create_card(
                "GPC Graphing",
                "Create and customize GPC chromatograms with various visualization options",
                "📈",
                "gpc_graphing",
                disabled=True,
                badge="Coming soon",
            ),
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()
