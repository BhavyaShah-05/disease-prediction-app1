import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Advanced Disease Predictor", layout="centered")

data = {
    'Symptom1': ['Fever', 'Cough', 'Headache', 'Fever', 'Nausea', 'Cough', 'Headache', 'Fever',
                 'Fatigue', 'Sneezing', 'Fever', 'Sore Throat', 'Fever', 'Cough', 'Body Pain', 'Loss of Taste',
                 'Rash', 'Joint Pain', 'Chest Pain', 'Dizziness', 'Fever', 'Swelling', 'Confusion', 'Blurred Vision',
                 'Numbness', 'Wheezing', 'Anxiety', 'Excessive Thirst', 'Frequent Urination', 'Weight Loss',
                 'Abdominal Pain', 'Diarrhea', 'Vomiting', 'Indigestion', 'Blood in Stool'],
    'Symptom2': ['Cough', 'Fever', 'Nausea', 'Headache', 'Cough', 'Fatigue', 'Fever', 'Nausea',
                 'Body Pain', 'Sore Throat', 'Body Pain', 'Cough', 'Loss of Taste', 'Sore Throat', 'Fatigue', 'Fever',
                 'Itching', 'Swelling', 'Shortness of Breath', 'Nausea', 'Yellow Skin', 'Red Eyes', 'Memory Loss', 'Eye Pain',
                 'Tingling', 'Shortness of Breath', 'Sweating', 'Hunger', 'Thirst', 'Fatigue',
                 'Bloating', 'Fever', 'Dehydration', 'Heartburn', 'Constipation'],
    'Disease': ['Flu', 'Flu', 'Migraine', 'Migraine', 'Food Poisoning', 'Cold', 'Cold', 'Food Poisoning',
                'Malaria', 'Allergy', 'Typhoid', 'Allergy', 'COVID-19', 'COVID-19', 'Malaria', 'COVID-19',
                'Measles', 'Arthritis', 'Angina', 'Vertigo', 'Hepatitis', 'Conjunctivitis', 'Alzheimer\'s', 'Glaucoma',
                'Diabetes', 'Asthma', 'Panic Disorder', 'Diabetes', 'Diabetes', 'Hyperthyroidism',
                'IBS', 'Gastroenteritis', 'Gastroenteritis', 'GERD', 'Hemorrhoids']
}

df = pd.DataFrame(data)

# Encode
le_symptom = LabelEncoder()
le_disease = LabelEncoder()
df['Symptom1_enc'] = le_symptom.fit_transform(df['Symptom1'])
df['Symptom2_enc'] = le_symptom.fit_transform(df['Symptom2'])
df['Disease_enc'] = le_disease.fit_transform(df['Disease'])

X = df[['Symptom1_enc', 'Symptom2_enc']]
y = df['Disease_enc']

# Train models
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=3)

dt_model.fit(X, y)
rf_model.fit(X, y)
knn_model.fit(X, y)


prescriptions = {
    'Flu': "🛌 Rest well, stay hydrated, and consider OTC meds like ibuprofen. Antiviral medications if diagnosed early.",
    'Migraine': "💊 Take prescribed migraine medications and avoid light/sound triggers. Try a cool, dark room.",
    'Food Poisoning': "🥤 Drink fluids, eat bland food, and use anti-nausea meds if needed. Seek medical help if severe.",
    'Cold': "🌡️ Rest, fluids, vitamin C, and over-the-counter decongestants. Increase humidity and gargle salt water.",
    'Allergy': "🌼 Use antihistamines and avoid known allergens. Consider nasal sprays and allergy shots for severe cases.",
    'Malaria': "💉 Seek medical attention immediately; antimalarial drugs are needed. Complete the full course of medication.",
    'Typhoid': "💊 Antibiotics and proper hydration are essential. See a doctor and maintain strict hygiene.",
    'COVID-19': "😷 Isolate, rest, hydrate, and consult a doctor if symptoms worsen. Monitor oxygen levels if possible.",
    'Measles': "🧪 Vitamin A supplements, isolation, rest, and fever reducers. Vaccination is the best prevention.",
    'Arthritis': "🩹 Anti-inflammatory medicines, physical therapy, and hot/cold treatments. Maintain healthy weight.",
    'Angina': "❤️ Nitroglycerin for immediate relief. Long-term: statins, aspirin, beta-blockers. Reduce stress.",
    'Vertigo': "🌀 Vestibular rehabilitation exercises, anti-dizziness medication, and careful movement.",
    'Hepatitis': "🏥 Rest, fluids, nutritious diet. Antiviral medications for some types. Avoid alcohol.",
    'Conjunctivitis': "👁️ Antibiotic drops for bacterial cases, cold compresses, and avoiding eye rubbing.",
    'Alzheimer\'s': "🧠 Cholinesterase inhibitors, memantine, regular routine and cognitive exercises.",
    'Glaucoma': "👓 Eye drops to reduce pressure, sometimes surgery or laser treatment. Regular check-ups.",
    'Diabetes': "💉 Blood sugar monitoring, insulin or oral medications, healthy diet, and regular exercise.",
    'Asthma': "🫁 Rescue and controller inhalers, avoiding triggers, and breathing exercises.",
    'Panic Disorder': "😌 Cognitive behavioral therapy, anti-anxiety medications, and breathing techniques.",
    'Hyperthyroidism': "⚕️ Anti-thyroid medications, radioactive iodine, or surgery in severe cases.",
    'IBS': "🚽 Dietary changes, stress management, and possibly fiber supplements or anti-spasmodic drugs.",
    'Gastroenteritis': "🦠 Fluids, electrolyte replacement, rest, and gradual return to normal diet.",
    'GERD': "🍽️ Proton pump inhibitors, H2 blockers, diet changes, and elevating head while sleeping.",
    'Hemorrhoids': "🧴 Over-the-counter creams, warm baths, high-fiber diet, and adequate hydration."
}

severity = {
    'Flu': 4,
    'Migraine': 5,
    'Food Poisoning': 6,
    'Cold': 3,
    'Allergy': 3,
    'Malaria': 8,
    'Typhoid': 7,
    'COVID-19': 7,
    'Measles': 6,
    'Arthritis': 5,
    'Angina': 8,
    'Vertigo': 5,
    'Hepatitis': 7,
    'Conjunctivitis': 2,
    'Alzheimer\'s': 9,
    'Glaucoma': 7,
    'Diabetes': 6,
    'Asthma': 6,
    'Panic Disorder': 5,
    'Hyperthyroidism': 6,
    'IBS': 4,
    'Gastroenteritis': 5,
    'GERD': 4,
    'Hemorrhoids': 3
}


recovery_time = {
    'Flu': 7,
    'Migraine': 2,
    'Food Poisoning': 3,
    'Cold': 7,
    'Allergy': 5,
    'Malaria': 14,
    'Typhoid': 21,
    'COVID-19': 14,
    'Measles': 10,
    'Arthritis': 0,  
    'Angina': 0,    
    'Vertigo': 7,
    'Hepatitis': 30,
    'Conjunctivitis': 7,
    'Alzheimer\'s': 0, 
    'Glaucoma': 0,      
    'Diabetes': 0,      # Chronic
    'Asthma': 0,        # Chronic
    'Panic Disorder': 0, # Chronic
    'Hyperthyroidism': 0, # Chronic
    'IBS': 0,           # Chronic
    'Gastroenteritis': 5,
    'GERD': 0,          # Chronic
    'Hemorrhoids': 14
}

st.title("🧬 Disease Prediction")
st.caption("Select symptoms and get a machine learning-based disease prediction with visual feedback")

# Input section
st.header("📋 Enter Symptoms")

# Model selection
model_choice = st.radio("Select Model", ['Decision Tree', 'Random Forest', 'K-Nearest Neighbors'])

# Symptom inputs
symptom_list = sorted(list(le_symptom.classes_))
col1, col2 = st.columns(2)
with col1:
    symptom1 = st.selectbox("Select First Symptom", symptom_list)
with col2:
    symptom2 = st.selectbox("Select Second Symptom", symptom_list)

# Show all diseases option
show_all_diseases = st.checkbox("Show All Possible Diseases", value=True)

# Predict button
if st.button("🔍 Predict Disease"):
    input_encoded = [[
        le_symptom.transform([symptom1])[0],
        le_symptom.transform([symptom2])[0]
    ]]
    
    # Predict based on selected model
    if model_choice == 'Decision Tree':
        model = dt_model
    elif model_choice == 'Random Forest':
        model = rf_model
    else:
        model = knn_model
    
    pred_encoded = model.predict(input_encoded)[0]
    pred_probs = model.predict_proba(input_encoded)[0]
    predicted_disease = le_disease.inverse_transform([pred_encoded])[0]
    
    # Prediction result
    st.subheader("🩺 Diagnosis")
    st.success(f"Based on your symptoms ({symptom1} and {symptom2}), you may have:\n")
    st.markdown(f"## **{predicted_disease}**")
    
    # Show prescription
    st.subheader("\n💊 Suggested Treatment")
    st.info(prescriptions.get(predicted_disease, "No prescription available."))
    
    # Disease severity and recovery info
    st.subheader("📋 Disease Information")
    
    # Severity gauge
    disease_severity = severity.get(predicted_disease, 5)
    st.write("Severity Level:")
    
    # Create a simple severity gauge
    severity_colors = ["#00cc96", "#00cc96", "#00cc96", "#ffa15a", "#ffa15a", 
                      "#ffa15a", "#ef553b", "#ef553b", "#ef553b", "#ef553b"]
    
    progress_html = f"""
    <div style="width:100%; background-color:#f0f0f0; border-radius:5px; height:30px">
        <div style="width:{disease_severity*10}%; background-color:{severity_colors[disease_severity-1]}; 
             height:30px; border-radius:5px; text-align:center; line-height:30px; color:white;">
            {disease_severity}/10
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)
    
    # Recovery time
    disease_recovery = recovery_time.get(predicted_disease, 0)
    if disease_recovery > 0:
        st.write(f"Expected Recovery Time: Approximately {disease_recovery} days")
    else:
        st.write("Expected Recovery Time: Chronic condition requiring ongoing management")
    
    # Build probabilities DataFrame
    prob_df = pd.DataFrame({
        'Disease': le_disease.inverse_transform(range(len(pred_probs))),
        'Probability': pred_probs
    })
    
    # Only show diseases with probability > 0 if showing all diseases
    if show_all_diseases:
        prob_df = prob_df[prob_df['Probability'] > 0]
    else:
        # Only show top 5
        prob_df = prob_df.sort_values(by="Probability", ascending=False).head(5)
    
    prob_df = prob_df.sort_values(by="Probability", ascending=False)
    
    # Top predictions table
    st.subheader("🔝 Differential Diagnosis")
    st.dataframe(prob_df.style.format({'Probability': '{:.1%}'}), use_container_width=True)
    
    # Bar chart with Plotly
    st.subheader("📊 Probability Chart")
    top_diseases = prob_df.head(8)
    
    fig = px.bar(
        top_diseases,
        y='Disease',
        x='Probability',
        orientation='h',
        color='Probability',
        color_continuous_scale='Bluyl',
        text_auto='.0%'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Additional visualizations
    st.header("📈 Additional Visualizations")
    
    # Radar chart of top 5 predictions
    st.subheader("Radar Chart - Top 5 Predictions")
    top5 = prob_df.head(5)
    
    # Prepare data for radar chart
    categories = top5['Disease'].tolist()
    values = top5['Probability'].tolist()
    
    # Close the loop for the radar chart
    categories.append(categories[0])
    values.append(values[0])
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    theta = np.linspace(0, 2*np.pi, len(categories))
    
    ax.plot(theta, values, 'o-', linewidth=2)
    ax.fill(theta, values, alpha=0.25)
    ax.set_thetagrids(theta * 180/np.pi, categories)
    ax.set_ylim(0, max(values) + 0.1)
    ax.grid(True)
    
    st.pyplot(fig)
    
    # Bubble chart
    st.subheader("Bubble Chart - Severity vs Recovery Time")
    
    # Get severity and recovery time for top diseases
    top_diseases_df = prob_df.head(5).copy()
    top_diseases_df['Severity'] = top_diseases_df['Disease'].map(lambda x: severity.get(x, 5))
    top_diseases_df['Recovery'] = top_diseases_df['Disease'].map(lambda x: recovery_time.get(x, 0))
    
    # Replace chronic conditions (0) with a high number for visualization
    top_diseases_df['Recovery'] = top_diseases_df['Recovery'].replace(0, 60)
    
    # Create bubble chart: x=Severity, y=Recovery, size=Probability
    fig2 = px.scatter(
        top_diseases_df,
        x="Severity",
        y="Recovery",
        size="Probability",
        color="Disease",
        hover_name="Disease",
        size_max=60,
        labels={"Recovery": "Recovery Time (days)", "Severity": "Severity (1-10)"}
    )
    
    # Replace y-axis label 60 with "Chronic"
    fig2.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=[10, 20, 30, 40, 50, 60],
            ticktext=['10', '20', '30', '40', '50', 'Chronic']
        )
    )
    
    st.plotly_chart(fig2, use_container_width=True)

    # Disease patterns section
    st.header("🔬 Disease Patterns")
    
    # Show common symptom combinations for predicted disease
    st.subheader(f"Common Symptom Pairs for {predicted_disease}")
    
    # Get all rows with the predicted disease
    disease_instances = df[df['Disease'] == predicted_disease]
    
    # Create a table of symptom combinations
    if not disease_instances.empty:
        symptom_pairs = pd.DataFrame({
            'Symptom 1': disease_instances['Symptom1'],
            'Symptom 2': disease_instances['Symptom2'],
        })
        st.table(symptom_pairs)
    else:
        st.write("No recorded symptom combinations for this disease.")
    
    # Show similar diseases based on symptom patterns
    st.subheader("Similar Diseases")
    
    # Get the predicted disease's symptoms
    disease_symptoms = set()
    for idx, row in df[df['Disease'] == predicted_disease].iterrows():
        disease_symptoms.add(row['Symptom1'])
        disease_symptoms.add(row['Symptom2'])
    
    # Count how many symptoms each disease shares with the predicted disease
    disease_similarity = {}
    for disease in df['Disease'].unique():
        if disease != predicted_disease:
            disease_symptoms_set = set()
            for idx, row in df[df['Disease'] == disease].iterrows():
                disease_symptoms_set.add(row['Symptom1'])
                disease_symptoms_set.add(row['Symptom2'])
            
            # Calculate Jaccard similarity
            intersection = len(disease_symptoms.intersection(disease_symptoms_set))
            union = len(disease_symptoms.union(disease_symptoms_set))
            similarity = intersection / union if union > 0 else 0
            
            disease_similarity[disease] = similarity
    
    # Convert to DataFrame and display
    similar_diseases = pd.DataFrame({
        'Disease': list(disease_similarity.keys()),
        'Symptom Similarity': list(disease_similarity.values())
    }).sort_values(by='Symptom Similarity', ascending=False).head(5)
    
    # Display as a horizontal bar chart
    fig3 = px.bar(
        similar_diseases,
        y='Disease',
        x='Symptom Similarity',
        orientation='h',
        color='Symptom Similarity',
        color_continuous_scale='Viridis',
        labels={'Symptom Similarity': 'Similarity Score'}
    )
    fig3.update_layout(height=300)
    st.plotly_chart(fig3, use_container_width=True)
