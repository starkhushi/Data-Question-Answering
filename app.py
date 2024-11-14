import streamlit as st
import pandas as pd
import google.generativeai as gemini
import re
import io

# Streamlit app
st.title('Data Question Answering App with Gemini')

# Input for Gemini API Key
gemini_api_key = st.text_input("Enter your Gemini API key:", type="password")

if gemini_api_key:
    # Set up Google API Key for Gemini
    gemini.configure(api_key=gemini_api_key)

    # Choose data input method
    input_method = st.radio("Choose your data input method:", ["Upload CSV File"])

    if input_method == "Upload CSV File":
        # Upload CSV file
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

        if uploaded_file is not None:
            # Load the CSV file into a DataFrame
            df = pd.read_csv(uploaded_file)
            st.write("Data preview:")
            st.write(df.head())

    # Only proceed if data (df) is available
    if 'df' in locals():
        # User input for question
        question = st.text_input("Ask a question about the data:")

        if st.button('Get Answer with Gemini'):
            if question:
                # Convert the DataFrame to a string to use as context
                context = df.to_string(index=False)
                model = gemini.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(f'Answer the following question based on the data: {context}\n\nQuestion: {question}')
                answer = response.text
                cleaned_response = re.sub(r'\*', '', answer)
                st.write(f"**Answer:** {cleaned_response}")

                # Provide download button for output only
                output = io.StringIO()
                output.write(f"Question: {question}\n\nAnswer:\n{cleaned_response}")
                st.download_button(
                    label="Download Answer as Text",
                    data=output.getvalue(),
                    file_name="answer_output.txt",
                    mime="text/plain"
                )
else:
    st.warning("Please enter your Gemini API key to use Gemini features.")
