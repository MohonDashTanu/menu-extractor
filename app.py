import base64
import json
import time
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# --- 1. DATA STRUCTURE DEFINITION ---
class AddOn(BaseModel):
    category: str | None = Field(default=None, description="e.g., 'Drinks Category', 'Extra Category', 'Sides'")
    name: str
    price: float | None = None

class MenuItem(BaseModel):
    category: str
    name: str
    description: str | None = None
    price: float | None = None
    add_ons: list[AddOn] = Field(default_factory=list)

class Menu(BaseModel):
    items: list[MenuItem]

# --- 2. STREAMLIT PAGE SETUP ---
st.set_page_config(layout="wide", page_title="AI Menu Extractor", page_icon="🍽️")
st.title("🍽️ AI Menu Extractor & Review")

# --- 3. CHECK FOR SECRETS ---
if "GEMINI_API_KEY" not in st.secrets:
    st.error("⚠️ GEMINI_API_KEY is not set in Streamlit Secrets!")
    st.stop()

api_key = st.secrets["GEMINI_API_KEY"]

# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Upload Menu")
    uploaded_file = st.file_uploader(
        "Upload a menu (PDF or Image)", 
        type=["pdf", "jpg", "jpeg", "png"]
    )

# --- 5. MAIN WORKSPACE ---
if uploaded_file:
    if "current_file" not in st.session_state or st.session_state["current_file"] != uploaded_file.name:
        st.session_state["current_file"] = uploaded_file.name
        st.session_state["menu_data"] = None

    # Left column (33%), Right column (66%)
    col_view, col_edit = st.columns([1, 2], gap="large")

    with col_view:
        st.subheader("Original Document")
        if uploaded_file.name.lower().endswith(".pdf"):
            base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
            pdf_embed = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700" type="application/pdf"></iframe>'
            st.markdown(pdf_embed, unsafe_allow_html=True)
        else:
            st.image(uploaded_file, use_container_width=True)

    with col_edit:
        st.subheader("Extracted Menu Items")

        if st.button("✨ Extract Menu with AI", type="primary", use_container_width=True):
            status_box = st.empty()
            status_box.info("Analyzing layout, items, and add-ons...")
            
            client = genai.Client(api_key=api_key)
            mime_type = "application/pdf" if uploaded_file.name.lower().endswith(".pdf") else "image/jpeg"
            
            # Explicit prompt telling AI to associate add-ons to specific items
            prompt = (
                "Extract all menu items. Identify descriptions and base prices. "
                "Critically, ensure you correctly associate each add-on with the specific menu item it belongs to. "
                "Group add-ons by their specific category (e.g., 'drinks category', 'extra category')."
            )
            
            max_retries = 100
            response = None
            
            # Visual Countdown Retry Logic
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=[
                            types.Part.from_bytes(data=uploaded_file.getvalue(), mime_type=mime_type),
                            prompt,
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=Menu,
                            temperature=0.1,
                        ),
                    )
                    status_box.empty() # Clear the status box on success
                    break 
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if "503" in error_msg or "429" in error_msg or "overloaded" in error_msg or "quota" in error_msg:
                        if attempt < max_retries - 1:
                            # Wait longer on each attempt: 3s, 5s, 7s...
                            wait_seconds = 1
                            for countdown in range(wait_seconds, 0, -1):
                                status_box.warning(f"Google servers are busy. Retrying in {countdown} seconds... (Attempt {attempt+1}/{max_retries})")
                                time.sleep(1)
                            status_box.info(f"Retrying now... (Attempt {attempt+1})")
                        else:
                            status_box.error("Servers are completely overloaded. Please wait a few minutes and click extract again.")
                            st.stop()
                    else:
                        status_box.error(f"Extraction failed: {e}")
                        st.stop()

            # Process Data into Parent/Child Rows
            if response:
                parsed = json.loads(response.text)
                rows = []
                
                for item in parsed.get("items", []):
                    # 1. Add the Main Item Row
                    rows.append({
                        "Category": item.get("category", "General"),
                        "Item Name": item.get("name", ""),
                        "Description": item.get("description") or "",
                        "Price": item.get("price") if item.get("price") is not None else None,
                        "Add-on Category": "",
                        "Add-on Name": "",
                        "Add-on Price": None,
                    })

                    # 2. Add a sub-row for every single Add-on that belongs to this item
                    for a in item.get("add_ons", []):
                        rows.append({
                            "Category": "",
                            "Item Name": "",
                            "Description": "",
                            "Price": None,
                            "Add-on Category": a.get("category") or "Add-on",
                            "Add-on Name": a.get("name", ""),
                            "Add-on Price": a.get("price") if a.get("price") is not None else None,
                        })

                st.session_state["menu_data"] = pd.DataFrame(rows)
                st.success("Extraction complete! You can edit, add, or delete rows below.")

        # Editable Data Table Display
        if st.session_state.get("menu_data") is not None:
            st.caption("Double-click any cell to edit. Scroll to bottom to add missing items (`+` button).")
            
            edited_df = st.data_editor(
                st.session_state["menu_data"],
                num_rows="dynamic",
                use_container_width=True,
                height=600,
            )

            st.download_button(
                label="📥 Download as CSV",
                data=edited_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_extracted.csv",
                mime="text/csv",
                use_container_width=True,
            )
else:
    st.info("👈 Upload a PDF or image in the sidebar to get started.")
