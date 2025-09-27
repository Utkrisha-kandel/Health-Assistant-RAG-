import os
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import create_vector as cv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage

PDF_DIR = "documents"

pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
patient_names = ["-- Select a patient --"] + [os.path.splitext(f)[0] for f in pdf_files]

st.set_page_config(page_title="Patient Records", layout="wide")
st.title("Health Assistant")

st.sidebar.header("Select Patient")
selected_patient = st.sidebar.selectbox("Choose a patient:", patient_names)

if selected_patient == "-- Select a patient --":
    st.markdown(
        """
        <h2 style="
            color: #16A085;
            font-family: 'Arial';
            font-weight: 700;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 10px;
        ">
         Select a Patient
        </h2>
        """,
        unsafe_allow_html=True
    )

else:
    pdf_path = os.path.join(PDF_DIR, f"{selected_patient}.pdf")
    st.subheader(f"📄 Full Report for: {selected_patient}")

    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.markdown("📑 PDF Preview:")
        pdf_viewer(input=pdf_bytes, width=900, height=700)
    else:
        st.warning("PDF not found for this patient.")

    user_query = st.text_input(f"Enter your query about {selected_patient}")
    submit_button = st.button("Submit")

    if submit_button and user_query:
        vector = cv.embed_text(user_query)

        vector_search_response = cv.vector_index.query(
            vector=vector,
            top_k=3,
            include_metadata=True,
            filter={"patient_name": selected_patient}
        )

        similar_texts = ""
        for match in vector_search_response["matches"]:
            text = match["metadata"]["text"]
            similar_texts += text + "\n\n"

        system_prompt = f"""You are a medical AI assistant acting as a doctor. 
Your task is to answer patient queries by analyzing their current health issue and 
retrieving relevant information from their past medical history stored in PDF documents.

Instructions:
1. Consider the patient's present symptoms or health concern.
2. Search through the retrieved history (only from {selected_patient}) for any past conditions, treatments, or patterns related to the current issue.
3. Provide a clear, concise, and professional explanation connecting the current issue to the past medical history.
4. If no relevant information is found in the history, explicitly state that no correlation is detected.
5. Always prioritize accuracy, clarity, and patient safety in your responses.
6. Reference relevant details from the PDFs when applicable, but do not include unnecessary text.

Relevant history:
{similar_texts}
"""

        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", api_key=os.getenv("API_KEY_GOOGLE"))

        messages = [
            [SystemMessage(content=system_prompt), HumanMessage(content=user_query)]
        ]

        llm_response = llm.generate(messages)
        llm_answer = llm_response.generations[0][0].text

        st.markdown(f"**Health-Assistant:** {llm_answer}")
