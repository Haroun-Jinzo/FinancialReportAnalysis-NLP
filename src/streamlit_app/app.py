import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
from pathlib import Path

# Get the project root (two levels up from streamlit_app/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Set working directory to project root
os.chdir(project_root)

# Add both project root and src directory to sys.path
src_dir = os.path.join(project_root, 'src')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import your actual backend logic directly
from agents.document_agent import DocumentAgent
from agents.analysis_agent import AnalysisAgent
from agents.query_agent import QueryAgent

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Financial AI Analyst",
    page_icon="💰",
    layout="wide"
)

# --- CSS FOR BETTER UI ---
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stAlert { margin-top: 1rem; }
    .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- CACHED AGENT LOADING ---
# @st.cache_resource ensures we only load heavy AI models ONCE, not on every click
@st.cache_resource
def load_agents():
    print("🔄 Loading AI Agents and Models...")
    # Ensure storage directory exists
    os.makedirs("data/documents", exist_ok=True)
    
    # Initialize agents
    doc_agent = DocumentAgent(storage_path="data/documents")
    ana_agent = AnalysisAgent()
    # Query agent needs references to the other two
    qry_agent = QueryAgent(document_agent=doc_agent, analysis_agent=ana_agent)
    
    return doc_agent, ana_agent, qry_agent

# Load the agents
try:
    with st.spinner("Starting AI Engines... (This takes 10-20 seconds only once)"):
        document_agent, analysis_agent, query_agent = load_agents()
except Exception as e:
    st.error(f"Failed to load AI models. Check your logs. Error: {e}")
    st.stop()

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    st.title("💰 Fin-AI Analyst")
    st.divider()
    
    st.subheader("1. Upload Report")
    uploaded_file = st.file_uploader("Upload PDF/DOCX", type=['pdf', 'docx', 'txt'])
    
    # Session State management for multiple documents
    if 'documents' not in st.session_state:
        st.session_state.documents = {}  # {doc_id: document_object}
    if 'current_doc_id' not in st.session_state:
        st.session_state.current_doc_id = None
    
    if uploaded_file:
        # Create documents directory with absolute path
        docs_dir = Path(project_root) / "data" / "documents"
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file locally with absolute path
        save_path = docs_dir / uploaded_file.name
        
        # Only process if it's a new file
        if not save_path.exists() or st.button("Process Document"):
            try:
                with st.status("Processing Document...", expanded=True) as status:
                    # 1. Save File
                    status.write("📂 Saving file...")
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Verify file was saved successfully
                    if not save_path.exists():
                        st.error(f"❌ Failed to save file: {save_path}")
                        st.stop()
                    
                    # 2. Ingest the document (returns Document object with ID)
                    status.write("📥 Ingesting document...")
                    doc_obj = document_agent.ingest_document(str(save_path))
                    
                    # 3. Process the document using its ID
                    status.write("🧠 Reading and analyzing text...")
                    doc = document_agent.process_document(doc_obj.id)
                    
                    # 4. Store in Session
                    st.session_state.documents[doc.id] = doc
                    st.session_state.current_doc_id = doc.id
                    status.update(label="✅ Processing Complete!", state="complete", expanded=False)
                    st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
                    
            except FileNotFoundError as e:
                st.error(f"❌ File error: {e}")
                st.info(f"File was saved to: {save_path}\nBut couldn't be read from there.")
            except ValueError as e:
                st.error(f"❌ Error processing document: {e}")
                st.info("Please ensure the file format is valid (PDF, DOCX, or TXT)")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                import traceback
                st.write(traceback.format_exc())

    st.divider()
    st.info("Supported formats: PDF, DOCX. \nRecommended: Quarterly Earnings Reports.")
    
    # --- DOCUMENT SELECTOR ---
    if st.session_state.documents:
        st.divider()
        st.subheader("2. Select Document")
        doc_options = {doc_id: doc.filename for doc_id, doc in st.session_state.documents.items()}
        selected_doc_id = st.selectbox(
            "Choose a document to analyze:",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options[x],
            index=list(doc_options.keys()).index(st.session_state.current_doc_id) if st.session_state.current_doc_id in doc_options else 0
        )
        st.session_state.current_doc_id = selected_doc_id
        
        # Show document info
        current_doc = st.session_state.documents[selected_doc_id]
        st.caption(f"📄 ID: {current_doc.id}\n\n⏰ Processed: {current_doc.processed_at.strftime('%Y-%m-%d %H:%M') if current_doc.processed_at else 'N/A'}")
        
        # Delete document option
        if st.button("🗑️ Delete This Document"):
            del st.session_state.documents[selected_doc_id]
            if st.session_state.documents:
                st.session_state.current_doc_id = list(st.session_state.documents.keys())[0]
            else:
                st.session_state.current_doc_id = None
            st.rerun()

# --- MAIN PAGE ---
if not st.session_state.current_doc_id or st.session_state.current_doc_id not in st.session_state.documents:
    st.markdown("## 👋 Welcome to Financial AI Analyst")
    st.markdown("Please upload a financial document (e.g., Apple Q3 Report) in the sidebar to begin.")
else:
    doc = st.session_state.documents[st.session_state.current_doc_id]
    
    # Top Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"Analysis: {doc.filename}")
        st.caption(f"Processed ID: {doc.id} | Status: {doc.status}")
    with col2:
        st.metric("Risk Level", "Calculating...", delta_color="inverse")

    # Tabs for different views
    tab_metrics, tab_risk, tab_trends, tab_compare, tab_chat, tab_raw = st.tabs(["📊 Key Metrics", "⚠️ Risk Analysis", "📈 Trends", "🔄 Compare", "💬 AI Chat", "📄 Raw Text"])

    # === TAB 1: METRICS ===
    with tab_metrics:
        st.subheader("Financial Highlights")
        
        if st.button("Extract Metrics", type="primary"):
            with st.spinner("Extracting financial data..."):
                metrics = analysis_agent.metric_extractor.extract_all_metrics(doc.text, "Current Period")
                
                # Display Metrics in columns
                if metrics:
                    # Income Statement
                    st.markdown("### 💵 Income Statement")
                    cols = st.columns(3)
                    
                    # Helper to find metric safely
                    def get_val(cat, name):
                        found = [m for m in metrics.get(cat, []) if name in m.name.lower()]
                        return found[0] if found else None

                    rev = get_val('income_statement', 'revenue')
                    net = get_val('income_statement', 'net_income')
                    eps = get_val('income_statement', 'eps')

                    if rev: cols[0].metric("Revenue", f"{rev.currency} {rev.value}{rev.unit}")
                    if net: cols[1].metric("Net Income", f"{net.currency} {net.value}{net.unit}")
                    if eps: cols[2].metric("EPS", f"{eps.value}")
                    
                    # Margins
                    st.markdown("### 📉 Margins & Ratios")
                    m_cols = st.columns(3)
                    gross = get_val('margin', 'gross')
                    op = get_val('margin', 'operating')
                    
                    if gross: m_cols[0].metric("Gross Margin", f"{gross.value}%")
                    if op: m_cols[1].metric("Operating Margin", f"{op.value}%")
                else:
                    st.warning("No standard metrics found. Try the Chat tab to ask specific questions.")

    # === TAB 2: RISKS ===
    with tab_risk:
        st.subheader("Risk Assessment")
        
        if st.button("Run Risk Scan"):
            with st.spinner("Scanning document for risks..."):
                risks = analysis_agent.analyze_risks(doc.text)
                
                # Risk Score Visualization
                level = risks.get('risk_level', 'UNKNOWN')
                st.info(f"Overall Risk Assessment: **{level}**")
                
                # Display individual risks
                for r in risks.get('risks', []):
                    with st.expander(f"🔴 {r['category']} ({r['severity']})"):
                        st.write(f"**Issue:** {r['description']}")
                        st.caption(f"Confidence: {r['confidence']:.0%}")

    # === TAB 3: TRENDS ===
    with tab_trends:
        st.subheader("Trend Analysis")
        
        if len(st.session_state.documents) < 2:
            st.warning("⚠️ Upload at least 2 documents to analyze trends")
        else:
            # Select documents for trend analysis
            trend_docs = st.multiselect(
                "Select documents to analyze trends:",
                options=list(st.session_state.documents.keys()),
                format_func=lambda x: st.session_state.documents[x].filename,
                default=[st.session_state.current_doc_id]
            )
            
            if len(trend_docs) < 2:
                st.info("Select 2 or more documents to see trends")
            elif st.button("Analyze Trends", type="primary"):
                with st.spinner("Calculating trends..."):
                    try:
                        # Prepare documents for analysis
                        documents_for_analysis = []
                        for doc_id in trend_docs:
                            doc_obj = st.session_state.documents[doc_id]
                            documents_for_analysis.append({
                                'period': doc_obj.metadata.get('period', doc_obj.filename),
                                'text': doc_obj.text
                            })
                        
                        # Run trend analysis
                        trends = analysis_agent.analyze_trends(documents=documents_for_analysis)
                        
                        if trends:
                            # Display trends
                            st.markdown("### 📊 Key Trends Identified")
                            if isinstance(trends, dict):
                                for trend_type, trend_data in trends.items():
                                    with st.expander(f"📈 {trend_type}"):
                                        st.write(trend_data)
                            else:
                                st.write(trends)
                        else:
                            st.info("No significant trends found in selected documents")
                    except Exception as e:
                        st.error(f"Error analyzing trends: {e}")

    # === TAB 4: COMPARE ===
    with tab_compare:
        st.subheader("Document Comparison")
        
        if len(st.session_state.documents) < 2:
            st.warning("⚠️ Upload at least 2 documents to compare")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                doc1_id = st.selectbox(
                    "Document 1:",
                    options=list(st.session_state.documents.keys()),
                    format_func=lambda x: st.session_state.documents[x].filename,
                    key="doc1"
                )
            
            with col2:
                doc2_id = st.selectbox(
                    "Document 2:",
                    options=list(st.session_state.documents.keys()),
                    format_func=lambda x: st.session_state.documents[x].filename,
                    key="doc2"
                )
            
            if doc1_id == doc2_id:
                st.warning("Please select two different documents")
            elif st.button("Compare Documents", type="primary"):
                with st.spinner("Comparing documents..."):
                    try:
                        doc1 = st.session_state.documents[doc1_id]
                        doc2 = st.session_state.documents[doc2_id]
                        
                        doc1_data = {
                            'name': doc1.metadata.get('company', doc1.filename),
                            'text': doc1.text,
                            'period': doc1.metadata.get('period')
                        }
                        
                        doc2_data = {
                            'name': doc2.metadata.get('company', doc2.filename),
                            'text': doc2.text,
                            'period': doc2.metadata.get('period')
                        }
                        
                        # Run comparison
                        comparison = analysis_agent.compare_documents(doc1_data, doc2_data, 'comprehensive')
                        
                        if comparison:
                            st.markdown("### 🔄 Comparison Results")
                            if isinstance(comparison, dict):
                                for section, content in comparison.items():
                                    with st.expander(f"📋 {section}"):
                                        st.write(content)
                            else:
                                st.write(comparison)
                        else:
                            st.info("No comparison data generated")
                    except Exception as e:
                        st.error(f"Error comparing documents: {e}")

    # === TAB 5: CHAT ===
    with tab_chat:
        st.subheader("Ask questions about this report")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat Input MUST be outside of tabs/columns/expanders
    if st.session_state.current_doc_id and st.session_state.current_doc_id in st.session_state.documents:
        if prompt := st.chat_input("Ex: What is the revenue growth compared to last year?"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = query_agent.query(prompt)
                        answer = response.get('answer', 'Sorry, I could not generate an answer.')
                        st.markdown(answer)
                        # Add assistant message
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error processing query: {e}")

    # === TAB 6: RAW TEXT ===
    with tab_raw:
        st.text_area("Extracted Text", doc.text, height=600)