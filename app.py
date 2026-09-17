import base64
import json
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# --- 1. DATA STRUCTURE DEFINITION ---
class AddOn(BaseModel):
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
    st.error(
        "⚠️ GEMINI_API_KEY is not set in Streamlit Secrets! "
        "Please go to App Settings > Secrets and add: GEMINI_API_KEY = \"your_api_key_here\""
    )
    st.stop()

api_key = st.secrets["GEMINI_API_KEY"]


# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Upload Menu")
    uploaded_file = st.file_uploader(
        "Upload a menu (PDF or Image)", 
        type=["pdf", "jpg", "jpeg", "png"]
    )
    st.caption("Supported formats: PDF, PNG, JPG, JPEG")


# --- 5. MAIN WORKSPACE ---
if uploaded_file:
    # Reset extracted data if the user switches to a new file
    if "current_file" not in st.session_state or st.session_state["current_file"] != uploaded_file.name:
        st.session_state["current_file"] = uploaded_file.name
        st.session_state["menu_data"] = None

    # Split into two equal columns
    col_view, col_edit = st.columns([1, 1], gap="medium")

    # LEFT COLUMN: Original Document Viewer
    with col_view:
        st.subheader("Original Document")
        if uploaded_file.name.lower().endswith(".pdf"):
            base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
            pdf_embed = (
                f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                f'width="100%" height="700" type="application/pdf"></iframe>'
            )
            st.markdown(pdf_embed, unsafe_allow_html=True)
        else:
            st.image(uploaded_file, use_container_width=True)

    # RIGHT COLUMN: Extraction & Interactive Table
    with col_edit:
        st.subheader("Extracted Menu Items")

        # Extraction trigger button
        if st.button("✨ Extract Menu with AI", type="primary", use_container_width=True):
            with st.spinner("Analyzing layout, items, and add-ons..."):
                try:
                    client = genai.Client(api_key=api_key)
                    
                    mime_type = "application/pdf" if uploaded_file.name.lower().endswith(".pdf") else "image/jpeg"

                    prompt = (
                        "Extract all menu items and group them by category. "
                        "Identify descriptions, base prices, and any add-ons/customizations with their additional prices."
                    )

                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
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

                    parsed = json.loads(response.text)
                    
                    # Flatten into table-friendly structure
                    rows = []
                    for item in parsed.get("items", []):
                        # Format add-ons into a readable string: "Extra Bacon ($2.00), Cheese ($1.00)"
                        addons_list = [
                            f"{a.get('name')} (+${a.get('price')})" if a.get("price") else a.get("name", "")
                            for a in item.get("add_ons", [])
                        ]
                        addons_str = "; ".join(filter(None, addons_list))

                        rows.append({
                            "Category": item.get("category", "General"),
                            "Item Name": item.get("name", ""),
                            "Description": item.get("description") or "",
                            "Price": item.get("price") if item.get("price") is not None else 0.0,
                            "Add-ons": addons_str,
                        })

                    st.session_state["menu_data"] = pd.DataFrame(rows)
                    st.success("Extraction complete! You can edit, add, or delete rows below.")

                except Exception as e:
                    st.error(f"Extraction failed: {e}")

        # Display Editable Table
        if st.session_state.get("menu_data") is not None:
            st.write("Double-click any cell to edit. Scroll to bottom to add missing items (`+` button).")
            
            # num_rows="dynamic" lets users add and delete rows manually
            edited_df = st.data_editor(
                st.session_state["menu_data"],
                num_rows="dynamic",
                use_container_width=True,
                height=520,
            )

            # Export button
            st.download_button(
                label="📥 Download as CSV",
                data=edited_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_extracted.csv",
                mime="text/csv",
                use_container_width=True,
            )
else:
    st.info("👈 Upload a PDF or image in the sidebar to get started.")
