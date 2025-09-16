import streamlit as st
from recommender import recommend_career
import pandas as pd
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
st.set_page_config(page_title='Ai Career Recommender',layout='centered')
st.title('AI Career Recommender')
st.write('Enter your skills and interested subjects(comma-seperated).The AI will suggest career and create pdf report')
skills=st.text_input('Enter your skills (comma-seperated)','Python,Communication,Marketing')
subjects=st.text_input('Enter interested subjects (comma-seperated)','Math,Physics,Biology,Chemistry')
num=st.slider('Number of recommendation',3,10,5)
if st.button('Recommend'):
    with st.spinner('Thinking...'):
        recs=recommend_career(skills,subjects,top_k=num)
    df_rec=pd.DataFrame(recs)
    st.subheader('Recommendation')
    st.dataframe(df_rec[['Career','Category','Match%']])
    st.subheader('Match percentage')
    chart_data=pd.DataFrame({'Match%':df_rec['Match%'].astype(float).values},index=df_rec['Career'].values)
    st.bar_chart(chart_data)
    for r in recs:
        with st.expander(f"{r['Career']}-{r['Match%']}%"):
            st.write(f"**Category:**{r['Category']}")
            st.write(f"**Profile:**{r['Profile']}")
            st.write('**suggested next steps:**')
            st.write('-take online course')
