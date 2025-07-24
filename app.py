import streamlit as st
import altair as alt
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import io
from reportlab.pdfgen import canvas
from track_utils import (
    create_page_visited_table,
    add_page_visited_details,
    view_all_page_visited_details,
    add_prediction_details,
    view_all_prediction_details,
    create_emotionclf_table,
    IST
)

# Load model
pipe_lr = joblib.load(open("./models/emotion_classifier_pipe_lr.pkl", "rb"))

# Prediction functions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Emoji dictionary
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🪮", "fear": "😨😱", "happy": "🫷",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}

# PDF generator
def generate_pdf(text, emotion, confidence):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    c.setFont("Helvetica", 12)
    c.drawString(100, 780, "Emotion Detection Report")
    c.drawString(100, 750, f"Text: {text[:60]}{'...' if len(text) > 60 else ''}")
    c.drawString(100, 720, f"Emotion: {emotion}")
    c.drawString(100, 690, f"Confidence: {round(confidence * 100, 2)}%")
    c.save()
    buffer.seek(0)
    return buffer

# Main app
def main():
    st.set_page_config(page_title="Emotion Detector", page_icon="💬", layout="centered")

    st.markdown("<h1 style='text-align: center; color: #6c63ff;'>💬 Emotion Detection in Text</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: grey;'>Analyze emotions like joy, anger, sadness, and more.</h4>", unsafe_allow_html=True)
    st.write("")

    menu = ["Home", "File Analysis", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    create_page_visited_table()
    create_emotionclf_table()

    if choice == "Home":
        add_page_visited_details("Home", datetime.now(IST))
        st.subheader("Try typing some text below:")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Your Text", height=150, placeholder="Enter your message here...")
            submit_text = st.form_submit_button(label='Analyze Emotion')

        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            add_prediction_details(raw_text, prediction, np.max(probability), datetime.now(IST))

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict.get(prediction, "")
                st.write(f"{prediction.capitalize()} {emoji_icon}")
                st.markdown(f"**Confidence:** {round(np.max(probability)*100, 2)}%")

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["Emotions", "Probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x=alt.X('Emotions', sort='-y'),
                    y='Probability',
                    color='Emotions'
                )
                st.altair_chart(fig, use_container_width=True)

            # Export Section
            st.markdown("---")
            st.subheader("📤 Export Results")

            # CSV Export
            csv_data = pd.DataFrame({
                "Text": [raw_text],
                "Predicted Emotion": [prediction],
                "Confidence": [round(np.max(probability)*100, 2)]
            })
            csv = csv_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download as CSV", data=csv, file_name="emotion_result.csv", mime='text/csv')

            # PDF Export
            pdf_buffer = generate_pdf(raw_text, prediction, np.max(probability))
            st.download_button("Download as PDF", data=pdf_buffer, file_name="emotion_report.pdf", mime='application/pdf')

    elif choice == "File Analysis":
        add_page_visited_details("File Analysis", datetime.now(IST))
        st.subheader("📁 Upload Text File for Emotion Analysis")
        uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])

        if uploaded_file is not None:
            text_lines = uploaded_file.read().decode("utf-8").splitlines()
            text_lines = [line.strip() for line in text_lines if line.strip()]

            results = []
            for line in text_lines:
                emotion = predict_emotions(line)
                proba = get_prediction_proba(line)
                confidence = np.max(proba)
                results.append({
                    "Text": line,
                    "Predicted Emotion": emotion,
                    "Emoji": emotions_emoji_dict.get(emotion, ""),
                    "Confidence": round(confidence * 100, 2)
                })

            df_result = pd.DataFrame(results)
            st.dataframe(df_result)

            # 📊 Emotion Count Chart (Fixed)
            st.markdown("### 📊 Emotion Distribution")
            emotion_counts = df_result["Predicted Emotion"].value_counts().reset_index()
            emotion_counts.columns = ["Emotion", "Count"]

            fig = px.bar(
                emotion_counts,
                x="Emotion", y="Count",
                labels={"Emotion": "Emotion", "Count": "Count"},
                title="Emotion Frequency",
                color="Emotion"
            )
            st.plotly_chart(fig)

            # 📥 CSV Download
            csv = df_result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name='emotion_analysis.csv',
                mime='text/csv',
            )

    elif choice == "Monitor":
        add_page_visited_details("Monitor", datetime.now(IST))
        st.subheader("Monitor Activity")

        with st.expander("📊 Page Metrics"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Page Name', 'Time of Visit'])
            st.dataframe(page_visited_details)

            pg_count = page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x='Page Name', y='Counts', color='Page Name')
            st.altair_chart(c, use_container_width=True)

            p = px.pie(pg_count, values='Counts', names='Page Name')
            st.plotly_chart(p, use_container_width=True)

        with st.expander('📈 Emotion Classifier Logs'):
            df_emotions = pd.DataFrame(view_all_prediction_details(), columns=['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
            st.dataframe(df_emotions)

            st.markdown("### 📊 Emotion Prediction Summary")
            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')

            bar_chart = alt.Chart(prediction_count).mark_bar().encode(
                x=alt.X('Prediction', sort='-y'),
                y='Counts',
                color='Prediction'
            ).properties(width=600)

            st.altair_chart(bar_chart, use_container_width=True)

            pie_chart = px.pie(
                prediction_count,
                values='Counts',
                names='Prediction',
                title='Emotion Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(pie_chart, use_container_width=True)

    else:
        add_page_visited_details("About", datetime.now(IST))

        st.subheader("📚 About This App")
        st.markdown("""
        Welcome to the **Emotion Detection in Text App**!  
        This app uses NLP and machine learning to detect emotional signals in text data.
        """)

        st.subheader("🚀 How It Works")
        st.markdown("""
        - You input your text.
        - The system extracts features and applies a trained model.
        - It predicts the most likely emotion and shows the confidence.
        """)

        st.subheader("🌟 Key Features")
        st.markdown("""
        - ✅ Real-time emotion detection  
        - 📊 Confidence score for predictions  
        - 🌟 Interactive visualizations  
        - 📁 Monitoring dashboard  
        """)

        st.subheader("🌟 Use Cases")
        st.markdown("""
        - Social media monitoring  
        - Customer feedback analysis  
        - Chatbot sentiment detection  
        - Market research  
        """)

if __name__ == '__main__':
    main()
