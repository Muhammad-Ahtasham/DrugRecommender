import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

class DrupRecommender():
    def __init__(self):
        self._svc = pickle.load(open("Model/svc.pkl", 'rb'))

        # load databasedataset===================================
        temp = {}
        df = pd.read_csv("Dataset/precautions_df.csv")
        df = df.to_numpy()
        for i in df:
            temp[i[1]] = [pre for pre in i[2:] if pre is not np.nan]
        self._pre = temp

        temp = {}
        df = pd.read_csv("Dataset/workout_df.csv")
        df = df.to_numpy()
        for i in df:
            temp[i[2]] = [pre for pre in i[3:] if pre is not np.nan]
        self._work = temp

        temp = {}
        df = pd.read_csv("Dataset/description.csv")
        df = df.to_numpy()
        for i in df:
            temp[i[0]] = [pre for pre in i[1:] if pre is not np.nan]
        self._des = temp        

        temp = {}
        df = pd.read_csv('Dataset/medications.csv')
        df = df.to_numpy()
        for i in df:
            temp[i[0]] = [pre for pre in eval(i[1]) if pre is not np.nan]
        self._med = temp 

        temp = {}
        df = pd.read_csv("Dataset/diets.csv")
        df = df.to_numpy()
        for i in df:
            temp[i[0]] = [pre for pre in eval(i[1]) if pre is not np.nan]
        self._diet = temp 

        self._sys = {
            'Itching': 0, 'Skin rash': 1, 'Nodal skin eruptions': 2, 'Continuous sneezing': 3, 'Shivering': 4, 'Chills': 5, 
            'Joint pain': 6, 'Stomach pain': 7, 'Acidity': 8, 'Ulcers on tongue': 9, 'Muscle wasting': 10, 'Vomiting': 11, 
            'Burning micturition': 12, 'Spotting urination': 13, 'Fatigue': 14, 'Weight gain': 15, 'Anxiety': 16, 
            'Cold hands and feet': 17, 'Mood swings': 18, 'Weight loss': 19, 'Restlessness': 20, 'Lethargy': 21, 
            'Patches in throat': 22, 'Irregular sugar level': 23, 'Cough': 24, 'High fever': 25, 'Sunken eyes': 26, 
            'Breathlessness': 27, 'Sweating': 28, 'Dehydration': 29, 'Indigestion': 30, 'Headache': 31, 'Yellowish skin': 32, 
            'Dark urine': 33, 'Nausea': 34, 'Loss of appetite': 35, 'Pain behind the eyes': 36, 'Back pain': 37, 'Constipation': 38, 
            'Abdominal pain': 39, 'Diarrhea': 40, 'Mild fever': 41, 'Yellow urine': 42, 'Yellowing of eyes': 43, 
            'Acute liver failure': 44, 'Fluid overload': 45, 'Swelling of stomach': 46, 'Swollen lymph nodes': 47, 
            'Malaise': 48, 'Blurred and distorted vision': 49, 'Phlegm': 50, 'Throat irritation': 51, 'Redness of eyes': 52, 
            'Sinus pressure': 53, 'Runny nose': 54, 'Congestion': 55, 'Chest pain': 56, 'Weakness in limbs': 57, 
            'Fast heart rate': 58, 'Pain during bowel movements': 59, 'Pain in anal region': 60, 'Bloody stool': 61, 
            'Irritation in anus': 62, 'Neck pain': 63, 'Dizziness': 64, 'Cramps': 65, 'Bruising': 66, 'Obesity': 67, 
            'Swollen legs': 68, 'Swollen blood vessels': 69, 'Puffy face and eyes': 70, 'Enlarged thyroid': 71, 
            'Brittle nails': 72, 'Swollen extremities': 73, 'Excessive hunger': 74, 'Extra-marital contacts': 75, 
            'Drying and tingling lips': 76, 'Slurred speech': 77, 'Knee pain': 78, 'Hip joint pain': 79, 'Muscle weakness': 80, 
            'Stiff neck': 81, 'Swelling joints': 82, 'Movement stiffness': 83, 'Spinning movements': 84, 'Loss of balance': 85, 
            'Unsteadiness': 86, 'Weakness of one body side': 87, 'Loss of smell': 88, 'Bladder discomfort': 89, 
            'Foul smell of urine': 90, 'Continuous feel of urine': 91, 'Passage of gases': 92, 'Internal itching': 93, 
            'Toxic look (typhoid)': 94, 'Depression': 95, 'Irritability': 96, 'Muscle pain': 97, 'Altered sensorium': 98, 
            'Red spots over body': 99, 'Belly pain': 100, 'Abnormal menstruation': 101, 'Dischromic patches': 102, 
            'Watering from eyes': 103, 'Increased appetite': 104, 'Polyuria': 105, 'Family history': 106, 'Mucoid sputum': 107, 
            'Rusty sputum': 108, 'Lack of concentration': 109, 'Visual disturbances': 110, 'Receiving blood transfusion': 111, 
            'Receiving unsterile injections': 112, 'Coma': 113, 'Stomach bleeding': 114, 'Distention of abdomen': 115, 
            'History of alcohol consumption': 116, 'Fluid overload (duplicate)': 117, 'Blood in sputum': 118, 
            'Prominent veins on calf': 119, 'Palpitations': 120, 'Painful walking': 121, 'Pus-filled pimples': 122, 
            'Blackheads': 123, 'Scarring': 124, 'Skin peeling': 125, 'Silver-like dusting': 126, 'Small dents in nails': 127, 
            'Inflammatory nails': 128, 'Blisters': 129, 'Red sore around nose': 130, 'Yellow crust oozing': 131
        }
        self.dise = {
            15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 
            33: 'Peptic ulcer disease', 1: 'AIDS', 12: 'Diabetes', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 
            23: 'Hypertension', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 
            29: 'Malaria', 8: 'Chickenpox', 11: 'Dengue', 37: 'Typhoid', 40: 'Hepatitis A', 19: 'Hepatitis B', 
            20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 
            10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemorrhoids (piles)', 18: 'Heart attack', 
            39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthritis', 
            5: 'Arthritis', 0: 'Vertigo (Paroxysmal Positional Vertigo)', 2: 'Acne', 38: 'Urinary tract infection', 
            35: 'Psoriasis', 27: 'Impetigo'
        }

    def helper(self, dis):
        desc = self._des[dis]

        pre = self._pre[dis]

        med = self._med[dis]

        diet = self._diet[dis]

        wrk = self._work[dis]

        print(desc, pre, med, diet, wrk)
        return (desc, pre, med, diet, wrk)

    def get_predicted_value(self, patient_symptoms):
        input_vector = np.zeros(len(self._sys))
        for item in patient_symptoms:
            input_vector[self._sys[item]] = 1
        dis = self.dise[self._svc.predict([input_vector])[0]]
        return dis

if 'dr' not in st.session_state:
    st.session_state['dr'] = DrupRecommender()
    st.session_state['options'] = st.session_state['dr']._sys.keys()

def main():

    st.title("Drug Recommender System")
    st.subheader("Efficiently find the right medication")

    st.write("This tool helps in recommending medications based on symptoms or conditions.")

    # Multi-option selector with a search bar
    selected_options = st.multiselect(
        'Select the Symptoms:',
        st.session_state['options'],  # list of options
        default=None  # no default selection
    )

    if st.button('Start Task'):
        # Show a spinner while the task is running
        with st.spinner('Processing...'):
            dis = st.session_state['dr'].get_predicted_value(selected_options)
            desc, pre, med, diet, wrk = st.session_state['dr'].helper(dis)
            desc = desc[0]
            pre = "\n".join([f"{i+1}. {j}" for i, j in enumerate(pre)])
            med = "\n".join([f"{i+1}. {j}" for i, j in enumerate(med)])
            diet = "\n".join([f"{i+1}. {j}" for i, j in enumerate(diet)])
            wrk = "\n".join([f"{i+1}. {j}" for i, j in enumerate(wrk)])
        
        # Show the result after the task is done
        st.header(dis)

        with st.expander(f"Description of {dis}", expanded=True):
            discription = st.empty()
            for i in range(len(desc)):
                discription.text(desc[:i+1])
                time.sleep(0.1)
        
        with st.expander(f"Precautions for {dis}", expanded=True):
            precautions = st.empty()
            for i in range(len(pre)):
                    precautions.text(pre[:i+1])
                    time.sleep(0.1)

        with st.expander(f"Medications for {dis}", expanded=True):
            precautions = st.empty()
            for i in range(len(med)):
                    precautions.text(med[:i+1])
                    time.sleep(0.1)
        
        with st.expander(f"Diet for {dis}", expanded=True):
            precautions = st.empty()
            for i in range(len(diet)):
                    precautions.text(diet[:i+1])
                    time.sleep(0.1)


        with st.expander(f"Workouts for {dis}", expanded=True):
            precautions = st.empty()
            for i in range(len(wrk)):
                    precautions.text(wrk[:i+1])
                    time.sleep(0.1)
if __name__ == "__main__":
    main()