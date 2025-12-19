import io
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None

try:  # Optional dependency for parsing Word documents
    import docx  # type: ignore
except ImportError:  # pragma: no cover - informs the UI instead
    docx = None

try:  # Optional dependency for real LLM calls
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - informs the UI instead
    OpenAI = None


MOCK_SIGNAL_FEED = [
    {
        "title": "Telehealth Platform Expansion",
        "publisher": "Midwest Integrated Care Network",
        "sector": "Healthcare",
        "location": "Illinois, USA",
        "budget": 4_500_000,
        "due_date": "2024-09-30",
        "summary": "Seeking HIPAA-compliant virtual care solution with remote monitoring integration.",
        "source": "https://signals.example.com/telehealth",
        "keywords": ["telehealth", "virtual care", "remote monitoring", "HIPAA"],
    },
    {
        "title": "Population Health Analytics Platform",
        "publisher": "State Department of Health",
        "sector": "Government",
        "location": "Ohio, USA",
        "budget": 7_200_000,
        "due_date": "2024-10-10",
        "summary": "Analytics modernization with interoperability to statewide HIE and quality registries.",
        "source": "https://signals.example.com/pophealth",
        "keywords": ["analytics", "HIE", "FHIR", "quality reporting"],
    },
    {
        "title": "Patient Access Modernization",
        "publisher": "Sunrise Health System",
        "sector": "Healthcare",
        "location": "Texas, USA",
        "budget": 2_000_000,
        "due_date": "2024-08-15",
        "summary": "Unified scheduling, call center, and digital front door platform for ambulatory clinics.",
        "source": "https://signals.example.com/patient-access",
        "keywords": ["patient access", "scheduling", "contact center", "digital front door"],
    },
    {
        "title": "Revenue Cycle Automation",
        "publisher": "Tri-County Medical Group",
        "sector": "Healthcare",
        "location": "California, USA",
        "budget": 3_100_000,
        "due_date": "2024-09-05",
        "summary": "AI-driven coding assistance, denial prevention, and payer contract intelligence.",
        "source": "https://signals.example.com/rcm",
        "keywords": ["revenue cycle", "automation", "coding", "payer"],
    },
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def authenticate(username: str, password: str) -> bool:
    """Very small demo authenticator; replace with your identity provider."""
    demo_users = {"demo": "copilot-demo", "architect": "welcome123"}
    return demo_users.get(username, "") == password


@st.cache_data(show_spinner=False)
def load_tabular_file(uploaded_file: io.BytesIO) -> pd.DataFrame:
    """Load CSV/Excel uploads into a DataFrame."""
    if uploaded_file is None:
        raise ValueError("Please upload a tabular knowledge base.")

    uploaded_file.seek(0)
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xls") or name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Please upload CSV or Excel.")


def call_llm(prompt: str, context: str = "", temperature: float = 0.2) -> str:
    """Call OpenAI if available, otherwise fall back to a simple template."""
    payload = f"{prompt}\n\nContext:\n{context}".strip()

    if OpenAI and "OPENAI_API_KEY" in st.secrets:
        try:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a product engineering copilot. Provide actionable, citeable answers.",
                    },
                    {"role": "user", "content": payload},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # pragma: no cover - UI notice only
            st.warning(f"Falling back to mock output because of an API issue: {exc}")

    # Offline fallback to keep the UI interactive without an API key.
    return (
        "Mock response (no API key detected):\n"
        f"- Summary: {prompt[:180]}...\n"
        "- Suggested sources: Upload your final knowledge base to cite.\n"
        "- Model reasoning: Available once an OpenAI key is configured."
    )


def find_relevant_rows(df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
    """Surface rows containing any of the provided keywords."""
    if df.empty or not keywords:
        return df.head(5)

    keyword_mask = pd.Series(False, index=df.index)
    text_columns = df.select_dtypes(include=["object", "string"])
    for word in keywords:
        if not word:
            continue
        contains_word = text_columns.apply(
            lambda col: col.str.contains(word.strip(), case=False, na=False)
        )
        keyword_mask = keyword_mask | contains_word.any(axis=1)
    matches = df[keyword_mask]
    return matches.head(10) if not matches.empty else df.head(5)


def read_docx(uploaded_file) -> Optional[str]:
    """Extract text from an uploaded .docx file."""
    if uploaded_file is None:
        return None
    if docx is None:
        st.error("python-docx is not installed. Run `pip install python-docx` to enable Word parsing.")
        return None
    uploaded_file.seek(0)
    document = docx.Document(uploaded_file)
    return "\n".join(para.text for para in document.paragraphs if para.text.strip())


def search_mock_signals(
    keywords: List[str], location: str = "", sector: str = "", min_budget: int = 0, max_budget: Optional[int] = None
) -> List[Dict[str, str]]:
    """Simple in-memory filter; swap with API integrations such as Crunchbase, GovWin, or internal catalogs."""
    results: List[Dict[str, str]] = []
    for record in MOCK_SIGNAL_FEED:
        if sector and record["sector"].lower() != sector.lower():
            continue
        if location and location.lower() not in record["location"].lower():
            continue
        if record["budget"] < min_budget:
            continue
        if max_budget and record["budget"] > max_budget:
            continue
        if keywords:
            record_keywords = " ".join(record["keywords"]).lower()
            if not any(word.lower() in record_keywords for word in keywords):
                continue
        results.append(record)
    return results


def fetch_live_signals(query: str, location: str = "", num_results: int = 10) -> List[Dict[str, str]]:
    """Call SerpAPI (or similar) to pull real web results. Requires SERPAPI_API_KEY in st.secrets."""
    if requests is None:
        st.error("The `requests` package is missing. Run `pip install requests` to enable live sourcing.")
        return []
    if "SERPAPI_API_KEY" not in st.secrets:
        st.warning("Add SERPAPI_API_KEY to st.secrets to enable live web sourcing.")
        return []

    params = {
        "q": query,
        "location": location or "United States",
        "num": num_results,
        "engine": "google",
        "api_key": st.secrets["SERPAPI_API_KEY"],
    }
    try:
        response = requests.get("https://serpapi.com/search.json", params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # pragma: no cover - runtime feedback only
        st.error(f"Live sourcing failed: {exc}")
        return []

    records: List[Dict[str, str]] = []
    organic_results = data.get("organic_results", [])
    for item in organic_results:
        summary = item.get("snippet") or item.get("title") or "No description provided."
        record = {
            "title": item.get("title") or "Untitled signal",
            "publisher": item.get("source") or "Unknown publisher",
            "sector": "Unknown",
            "location": location or "N/A",
            "budget": 0,
            "due_date": "TBD",
            "summary": summary,
            "source": item.get("link") or "",
            "keywords": query.split(),
        }
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Page implementations
# ---------------------------------------------------------------------------

def login_page():
    st.title("Product Engineering Copilot")
    st.caption("Prototype login — replace with SSO/Identity provider in production.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        if authenticate(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("Login successful!")
        else:
            st.error("Invalid credentials. Try demo / copilot-demo.")


def home_page():
    st.title("🧠 Product Engineering Copilot")
    st.write(
        """
        Central console for architects, PMs, and engineers to reason over specs, drafts, and research feeds.
        Use the navigation pane to move between capability modules.
        """
    )

    st.subheader("Included capabilities")
    st.markdown(
        """
        1. **Design Q&A** – answer architectural questions or batch-generate responses for backlog spreadsheets.
        2. **Requirements Miner** – ingest PRDs/RFCs and turn free-form language into structured requirements.
        3. **Pattern Workshop** – co-create reusable blueprints with conversational guidance.
        4. **Tech Radar** – monitor external feeds for trends, components, and reference implementations.
        5. **Readiness Scoring** – assess feature/initiative viability across architecture, compliance, and delivery risk.
        6. **Architecture Red-Team** – stress-test design drafts for clarity, compliance, and differentiation gaps.
        """
    )

    st.info(
        "All pages are LLM-ready. Configure your OpenAI API key via `st.secrets` to enable direct calls. "
        "Until then, mock responses keep the flow testable."
    )


def design_assistant_page():
    st.header("Design Q&A")
    st.caption("Answer architecture or feature questions individually or across backlog spreadsheets.")

    tab_single, tab_excel = st.tabs(["Single question answer", "Backlog spreadsheet"])

    with tab_single:
        st.subheader("Single Question Answering")
        kb_file = st.file_uploader("Architecture knowledge base (CSV/XLSX)", type=["csv", "xls", "xlsx"], key="single_kb")
        filters = st.text_input("Tags / components (comma separated)", help="Example: auth, HIPAA, eventing, latency")
        question = st.text_area("Engineering question", height=150)

        if st.button("Generate answer", type="primary"):
            if not question:
                st.warning("Provide a question to continue.")
            else:
                df = None
                context = ""
                try:
                    if kb_file:
                        df = load_tabular_file(kb_file)
                        keywords = [f.strip() for f in filters.split(",") if f.strip()]
                        relevant = find_relevant_rows(df, keywords)
                        context = relevant.to_csv(index=False)
                        st.write("**Top matching references**")
                        st.dataframe(relevant)
                except Exception as exc:
                    st.error(f"Failed to read knowledge base: {exc}")

                prompt = (
                    "Craft a clear engineering response that references existing components or decisions. "
                    "Highlight trade-offs, call out dependencies, and note risks or assumptions."
                    f"\n\nQuestion: {question}\nTags: {filters}"
                )
                answer = call_llm(prompt, context=context)
                st.write("### Suggested answer")
                st.write(answer)

    with tab_excel:
        st.subheader("Backlog answering")
        excel_file = st.file_uploader("Upload backlog Excel", type=["xls", "xlsx"], key="excel_upload")
        response_col_name = st.text_input("Response column name", value="Design_Response")

        if excel_file:
            try:
                df = load_tabular_file(excel_file)
                question_columns = df.columns.tolist()
                question_col = st.selectbox("Story / question column", question_columns)
                context_cols = st.multiselect(
                    "Optional context columns",
                    [col for col in question_columns if col != question_col],
                )
            except Exception as exc:
                df = None
                st.error(f"Could not read Excel file: {exc}")

            if df is not None and st.button("Generate spreadsheet responses"):
                output_df = df.copy()
                responses: List[str] = []
                for _, row in output_df.iterrows():
                    question_text = str(row.get(question_col, ""))
                    context_values = "\n".join(f"{col}: {row[col]}" for col in context_cols)
                    prompt = (
                        "Provide a concise design note for the following backlog entry. "
                        "Reference standards, APIs, or SLAs if present."
                        f"\n\nItem: {question_text}"
                        f"\nContext:\n{context_values}"
                    )
                    responses.append(call_llm(prompt, temperature=0.1))
                output_df[response_col_name] = responses
                st.success(f"Added `{response_col_name}` column with {len(responses)} entries.")
                st.dataframe(output_df.head(20))

                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    output_df.to_excel(writer, index=False)
                buffer.seek(0)
                st.download_button(
                    "Download responses",
                    data=buffer,
                    file_name="design_responses.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )


def requirements_miner_page():
    st.header("Requirements Miner")
    st.caption("Upload PRDs / RFCs or paste discovery notes to structure actionable requirements.")

    doc_file = st.file_uploader("Upload Word document (.docx)", type=["docx"])
    pasted_text = st.text_area("Or paste PRD / RFC text", height=220)
    instructions = st.text_area(
        "Extraction instructions",
        value="Group requirements by capability, state acceptance criteria, and flag unknowns or dependencies.",
        height=150,
    )

    if st.button("Extract requirements"):
        text = ""
        if doc_file:
            text = read_docx(doc_file) or ""
        if not text:
            text = pasted_text
        if not text:
            st.warning("Provide a document or pasted text to analyze.")
            return

        prompt = (
            "You are documenting product requirements for an engineering team.\n"
            f"{instructions}\n\nDocument text:\n{text[:4000]}"
            "\n\nReturn a markdown table with Requirement, Capability Group, Acceptance Criteria, and Open Questions columns."
        )
        extraction = call_llm(prompt, temperature=0.3)
        st.markdown("### Requirements draft")
        st.write(extraction)


def pattern_workshop_page():
    st.header("Pattern Workshop")
    st.caption("Iteratively refine design blueprints, runbooks, or spec templates with conversational guidance.")

    instructions = st.text_area(
        "Base instructions",
        "Create a reusable blueprint for building a multi-tenant API gateway with rate limiting and zero-downtime deploys.",
    )
    reference_files = st.file_uploader(
        "Reference materials (optional, multiple allowed)",
        type=["txt", "md", "docx"],
        accept_multiple_files=True,
    )

    reference_texts: List[str] = []
    for uploaded in reference_files or []:
        if uploaded.name.endswith(".docx"):
            text = read_docx(uploaded)
        else:
            uploaded.seek(0)
            text = uploaded.read().decode("utf-8")
        if text:
            reference_texts.append(f"--- {uploaded.name} ---\n{text}")

    st.divider()
    st.write("### Chat with the copilot")

    if "pattern_chat" not in st.session_state:
        st.session_state.pattern_chat = []

    for message in st.session_state.pattern_chat:
        st.chat_message(message["role"]).write(message["content"])

    user_prompt = st.chat_input("Ask for revisions, new sections, or trade-off analysis")
    if user_prompt:
        st.chat_message("user").write(user_prompt)
        st.session_state.pattern_chat.append({"role": "user", "content": user_prompt})

        context_blob = "\n\n".join(reference_texts)
        prompt = (
            f"{instructions}\n\nUser request: {user_prompt}\n\n"
            "Only return the requested content. Include architecture-ready headings, diagrams (described), and suggested tooling."
        )
        response = call_llm(prompt, context=context_blob, temperature=0.4)
        st.chat_message("assistant").write(response)
        st.session_state.pattern_chat.append({"role": "assistant", "content": response})


def tech_radar_page():
    st.header("Tech Radar")
    st.caption("Scan curated feeds and the open web for relevant architecture patterns or partner solutions.")

    with st.form("tech_radar"):
        keywords_text = st.text_input("Search keywords", value="event-driven telemetry, edge analytics")
        location = st.text_input("Preferred geography", value="USA")
        sector = st.selectbox("Domain", ["Any", "Healthcare", "Government", "Education"])
        min_budget = st.number_input("Minimum estimated investment (USD)", min_value=0, value=500000, step=50000)
        max_budget = st.number_input("Maximum estimated investment (USD, 0 = no limit)", min_value=0, value=4000000, step=50000)
        source_mode = st.selectbox(
            "Data source",
            [
                "Curated feed only",
                "Live web search (SerpAPI)",
                "Hybrid (live + curated)",
            ],
        )
        due_before = st.text_input(
            "Publish/due date before (YYYY-MM-DD, optional)",
            value="",
            help="Filter curated feed items whose timelines fit your planning window.",
        )
        intel_notes = st.text_area(
            "Research goals / watchlist tags",
            value="Focus on telemetry pipelines, HIPAA-aligned data platforms, or observability tooling partnerships.",
        )
        submitted = st.form_submit_button("Search signals", type="primary")

    if not submitted:
        st.info("Enter discovery criteria and click search to pull curated results.")
        return

    keywords = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]
    sector_filter = "" if sector == "Any" else sector
    max_budget_value = max_budget if max_budget > 0 else None
    hits: List[Dict[str, str]] = []

    if source_mode in {"Curated feed only", "Hybrid (live + curated)"}:
        hits.extend(search_mock_signals(keywords, location, sector_filter, int(min_budget), max_budget_value))

    if source_mode in {"Live web search (SerpAPI)", "Hybrid (live + curated)"}:
        query = " ".join(keywords) if keywords else "cloud architecture case study"
        live_hits = fetch_live_signals(query=query, location=location, num_results=10)
        hits.extend(live_hits)

    # Deduplicate by source URL/title to keep the list tidy
    unique_hits: List[Dict[str, str]] = []
    seen_keys = set()
    for record in hits:
        key = (record.get("source"), record.get("title"))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_hits.append(record)
    hits = unique_hits

    if due_before:
        try:
            due_cutoff = datetime.strptime(due_before, "%Y-%m-%d").date()
            filtered_hits = []
            for record in hits:
                due_text = record.get("due_date", "")
                try:
                    record_due = datetime.strptime(due_text, "%Y-%m-%d").date()
                except ValueError:
                    continue  # skip entries without a structured due date
                if record_due <= due_cutoff:
                    filtered_hits.append(record)
            hits = filtered_hits
        except ValueError:
            st.warning("Could not parse due date filter; showing all matches.")

    if not hits:
        st.warning("No signals matched the filters. Broaden criteria or update the data source.")
        return

    st.success(f"Found {len(hits)} relevant signals.")

    for idx, record in enumerate(hits):
        due_label = record["due_date"] if record["due_date"] != "TBD" else "n/a"
        with st.expander(f"{record['title']} – {record['publisher']} (Timeline {due_label})", expanded=idx == 0):
            st.write(f"**Domain:** {record['sector']} | **Location:** {record['location']}")
            st.write(f"**Estimated investment / scope:** ${record['budget']:,}")
            st.write(record["summary"])
            if record["source"]:
                st.markdown(f"[Source link]({record['source']})")
            st.caption(f"Indexed keywords: {', '.join(record['keywords'])}")

            if st.button("Generate research brief", key=f"sourcer_{idx}"):
                prompt = (
                    "You are advising a platform engineering team evaluating external signals.\n"
                    f"Signal: {record['title']} ({record['publisher']})\n"
                    f"Summary: {record['summary']}\n"
                    f"Scope: ${record['budget']:,}\n"
                    f"Location: {record['location']}\n"
                    f"Research goals: {intel_notes}\n"
                    f"Keywords: {', '.join(record['keywords'])}\n"
                    "Provide 1) Why this matters, 2) Potential integrations or reuse, 3) Suggested follow-ups."
                )
                capture_brief = call_llm(prompt, temperature=0.3)
                st.markdown("**Research notes**")
                st.write(capture_brief)

    if st.button("Summarize signal landscape"):
        context = "\n\n".join(f"{rec['title']} – {rec['summary']}" for rec in hits)
        summary_prompt = (
            "Summarize the common threads across these signals for a platform engineering team. "
            "Highlight dominant technical themes, regulatory patterns, and suggested evaluation next steps.\n\n"
            f"{context}"
        )
        overview = call_llm(summary_prompt, temperature=0.35)
        st.markdown("### Signal overview")
        st.write(overview)


def architecture_red_team_page():
    st.header("Architecture Red-Team")
    st.caption("Act as a skeptical reviewer to stress-test design documents before they hit review boards.")

    with st.form("red_team"):
        draft_file = st.file_uploader("Upload design doc (.docx or .txt)", type=["docx", "txt"])
        draft_text = st.text_area("Or paste design doc / ADR content", height=220)
        focus_areas = st.multiselect(
            "Focus areas",
            ["Scalability", "Security", "Observability", "Resilience", "Costs", "Team readiness"],
            default=["Scalability", "Security", "Resilience"],
        )
        tone = st.select_slider(
            "Critique tone",
            options=["Balanced", "Strict", "Brutal"],
            value="Strict",
            help="Brutal yields more aggressive critiques and lower tolerance for gaps.",
        )
        reviewer_instructions = st.text_area(
            "Reviewer instructions",
            value="Identify critical architectural risks, missing telemetry, compliance blockers, and unclear ownership. "
            "Suggest concrete edits, tests, or design alternatives.",
        )
        submitted = st.form_submit_button("Run red-team review", type="primary")

    if not submitted:
        return

    source_text = ""
    if draft_file is not None:
        if draft_file.name.endswith(".docx"):
            source_text = read_docx(draft_file) or ""
        else:
            draft_file.seek(0)
            source_text = draft_file.read().decode("utf-8")

    if not source_text:
        source_text = draft_text

    if not source_text:
        st.warning("Provide a document or paste content to critique.")
        return

    words = len(source_text.split())
    st.write(f"Draft length: **{words} words**")

    prompt = (
        "You are leading an architecture red-team review for a product engineering organization. "
        f"Critique tone: {tone}. Focus areas: {', '.join(focus_areas)}.\n"
        f"{reviewer_instructions}\n\nDraft:\n{source_text[:6000]}"
        "\n\nReturn:\n"
        "1. Summary judgment (ship / rework recommendation).\n"
        "2. Table of critical findings with Severity (High/Med/Low), Impact, and Fix.\n"
        "3. Annotated suggestions for key sections.\n"
        "4. Checklist of experiments, data, or decisions needed before approval."
    )
    critique = call_llm(prompt, temperature=0.25)
    st.markdown("### Red-team report")
    st.write(critique)


def readiness_scoring_page():
    st.header("Readiness Scoring")
    st.caption("Score a product initiative across architecture fit, security, and delivery risk to guide go/hold decisions.")

    with st.form("readiness_scoring"):
        opp_name = st.text_input("Initiative name", value="Telemetry Pipeline Refresh")
        summary = st.text_area("Initiative summary", height=120)
        requirements = st.text_area(
            "Key objectives / requirements",
            height=120,
            help="Paste highlights from the PRD, roadmap entry, or steering notes.",
        )
        priorities = st.multiselect(
            "Primary engineering priorities",
            [
                "Scalability",
                "Performance",
                "Security & compliance",
                "Developer productivity",
                "Time-to-market",
                "Cost efficiency",
            ],
            default=["Scalability", "Security & compliance"],
        )

        st.write("### Internal scoring (0 = poor, 5 = excellent)")
        fit_score = st.slider("Architecture fit with current platforms", 0.0, 5.0, 3.5, 0.5)
        differentiation_score = st.slider("Innovation / customer impact", 0.0, 5.0, 3.0, 0.5)
        compliance_score = st.slider("Security & regulatory readiness", 0.0, 5.0, 4.0, 0.5)
        relationship_score = st.slider("Team capacity & stakeholder alignment", 0.0, 5.0, 2.5, 0.5)
        risk_score = st.slider("Delivery risk (higher = more risk)", 0.0, 5.0, 1.5, 0.5)

        mitigation = st.text_area("Known risks & mitigation plan", height=100)

        submitted = st.form_submit_button("Score initiative", type="primary")

    if not submitted:
        return

    weights: Dict[str, float] = {
        "fit": 0.35,
        "differentiation": 0.2,
        "compliance": 0.25,
        "relationship": 0.2,
    }
    weighted_score = (
        fit_score * weights["fit"]
        + differentiation_score * weights["differentiation"]
        + compliance_score * weights["compliance"]
        + relationship_score * weights["relationship"]
    )
    penalty = risk_score * 0.08  # mild penalty per risk point
    normalized = max(0.0, min(5.0, weighted_score - penalty))
    readiness_percent = round((normalized / 5.0) * 100, 1)

    if readiness_percent >= 75:
        recommendation = "Green-light build – strong runway."
    elif readiness_percent >= 55:
        recommendation = "Conditional go – close gaps on staffing and compliance."
    elif readiness_percent >= 35:
        recommendation = "Monitor – mature architecture assumptions before resourcing."
    else:
        recommendation = "Defer – revisit after addressing major risks."

    st.metric("Estimated readiness", f"{readiness_percent}%")
    st.info(recommendation)
    st.write("**Score breakdown**")
    st.table(
        pd.DataFrame(
            {
                "Category": ["Architecture fit", "Innovation", "Security readiness", "Team alignment", "Risk penalty"],
                "Score": [
                    f"{fit_score}/5",
                    f"{differentiation_score}/5",
                    f"{compliance_score}/5",
                    f"{relationship_score}/5",
                    f"-{penalty:.2f}",
                ],
            }
        )
    )

    prompt = (
        "You are a principal engineer advising product leadership on initiative readiness.\n"
        f"Initiative: {opp_name}\n"
        f"Summary: {summary}\n"
        f"Key objectives: {requirements}\n"
        f"Priorities: {', '.join(priorities)}\n"
        f"Scores -> Architecture fit: {fit_score}/5, Innovation: {differentiation_score}/5, "
        f"Security readiness: {compliance_score}/5, Team alignment: {relationship_score}/5, "
        f"Risk: {risk_score}/5\n"
        f"Calculated readiness: {readiness_percent}%\n"
        f"Risks & mitigation: {mitigation}\n\n"
        "Provide: 1) Recommendation, 2) Critical technical or org gaps, 3) Next steps for the next 2 sprints."
    )
    advisory = call_llm(prompt, temperature=0.35)
    st.markdown("### Readiness advisory")
    st.write(advisory)


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Product Engineering Copilot", layout="wide")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None

    if not st.session_state.authenticated:
        login_page()
        return

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/3/3a/Logo_placeholder.png", caption="Company logo")
        st.write(f"Signed in as **{st.session_state.username}**")
        selection = st.radio(
            "Navigate",
            [
                "Home",
                "Design Q&A",
                "Requirements Miner",
                "Pattern Workshop",
                "Tech Radar",
                "Readiness Scoring",
                "Architecture Red-Team",
            ],
        )
        if st.button("Log out"):
            st.session_state.authenticated = False
            st.session_state.pop("pattern_chat", None)
            st.experimental_rerun()

    if selection == "Home":
        home_page()
    elif selection == "Design Q&A":
        design_assistant_page()
    elif selection == "Requirements Miner":
        requirements_miner_page()
    elif selection == "Pattern Workshop":
        pattern_workshop_page()
    elif selection == "Tech Radar":
        tech_radar_page()
    elif selection == "Readiness Scoring":
        readiness_scoring_page()
    else:
        architecture_red_team_page()


if __name__ == "__main__":
    main()
