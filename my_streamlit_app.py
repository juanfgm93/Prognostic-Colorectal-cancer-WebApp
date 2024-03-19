import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, mean_squared_error


# Set page configuration
st.set_page_config(layout="wide", page_title="Colorectal Cancer Prognostic App", page_icon=":hospital:")

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #C3DCFD;
    }
</style>
""", unsafe_allow_html=True)
# Load models
models = {}
model_files = ['logreg.pkl', 'knn.pkl', 'forest.pkl', 'xgb_reg.pkl']

# Mapping from model filenames to display names
model_display_names = {
    'logreg.pkl': 'Logistic Regression',
    'knn.pkl': 'K-Nearest Neighbors',
    'forest.pkl': 'Random Forest',
    'xgb_reg.pkl': 'XGBoost'
}

for model_file in model_files:
    with open(model_file, 'rb') as file:
        models[model_display_names[model_file]] = pickle.load(file)

# Load validation data (assuming you have it)
validation_data = pd.read_csv('validation_data.csv') 

def get_prediction(model, input_df):
    return model.predict(input_df)

def home():
    st.title('Colorectal cancer | Prognostic App')

    st.image('https://imgur.com/WUcvMwJ.jpg', use_column_width=True)
    st.write("""
            Our app is designed to predict patient prognosis for individuals affected by colorectal cancer. We utilize a combination of 
            demographic, clinical, and genomic data to build accurate predictive models.
             Colorectal cancer is a prevalent form of cancer that affects the colon or rectum. By analyzing various factors such as age, gender, 
             prior medical history, genetic mutations, and tumor characteristics, our models provide valuable insights into the likely prognosis of patients.
             """)
    st.write("""""")
    st.write("""""")
    st.write("""""")
    st.write("""""")
    st.write("""""")

    # Add text
    col1, col2 = st.columns([3, 7])  # Define columns
    with col1:
        st.write("""
            **Contributors**:
            - Juan F. García-Moreno
            - https://github.com/juanfgm93
        """)

    # Add some space
    st.write(" ")

    # Add image with alignment to the right
    with col2:
        st.image('https://imgur.com/OZh9uca.jpg', width=90)


def about_colorectal_cancer():
    
    st.title('Colorectal cancer in numbers')
    st.write("""
            Colorectal cancer is a disease in which cells in the colon or rectum grow out of control. Sometimes it is called colon cancer, for short. The colon is the large intestine or large bowel. The rectum is the passageway that connects the colon to the anus.
            Throughout this page, you'll find information about this disease, helping you understand the purpose of the predictive app.""")
    st.image('https://imgur.com/8RN35L8.jpg', width=600)
    
    st.header('Incidence rates | Top countries')
    st.write("""
            Colorectal cancer is the third most common cancer worldwide. The table below displays countries with the highest rates of colorectal cancer for both men and women in 2020¹.""")
    st.image('https://imgur.com/cnRlTS0.jpg', width=600)

    st.header('Colorectal cancer deaths')
    st.write("""
             Worldwide, colorectal cancer is the second leading cause of cancer death. In 2020, around **935,000 people died** from colorectal cancer. This includes 576,858 people with colon cancer and 339,022 people with rectal cancer.
             The survival rates for colorectal cancer vary based on several factors. These include the stage of cancer, a person’s age and general health, and how well the treatment plan works.
             The 5-year relative **survival rate** for colorectal cancer in the United States was 65%².
             The following table shows total **global colorectal cancer mortality** in 2020 across countries for both men and women¹.""")
    st.image('https://imgur.com/mDmFZTN.jpg', width=600)

    st.header('Main causes of colorectal cancer')

    st.write("""        
            Your risk of getting colorectal cancer increases as you get older. Other risk factors include having⁴:
            - **Inflammatory bowel disease** such as Crohn’s disease or ulcerative colitis.
            - **A personal or family history** of colorectal cancer or colorectal polyps.
            - A genetic syndrome such as **familial adenomatous polyposis** (FAP) or **Lynch syndrome**.
  
            Lifestyle factors that may increase risk of colorectal cancer include⁴:
            - Lack of regular **physical activity**.
            - A diet low in **fruit and vegetables**.
            - A low-fiber and high-fat diet, or a diet high in **processed meats**.
            - **Overweight** and **obesity**.
            - **Alcohol** consumption and **Tobacco** use.
            """)
    st.image('https://imgur.com/TeAyhDe.jpg', width=700)

    st.write("""
             References:
             1. https://www.wcrf.org/cancer-trends/colorectal-cancer-statistics/
             2. https://www.cancer.org/
             3. https://halifaxhealth.org/are-you-at-risk-for-colorectal-cancer/
             4. https://www.cdc.gov/
             """)
    
def data_source():
    
    st.title('Data source and visualization')
    st.write("""In this section, the data sources used to build the prognostic app are displayed. Data were obtained from the TCGA database. 
             Clinical and demographic data was retrieved using the Genomic Data Commons data portal, and genomic data was obtained using the cBioPortal website.
             """)
    st.header('The Cancer Genome Atlas - TCGA')
    st.image('https://imgur.com/PVbPDPY.jpg', width=900)
    st.write("""Click on this link to gain access to the site: https://portal.gdc.cancer.gov/
            """)

    st.header('cBioPortal for Cancer Genomics')
    st.image('https://imgur.com/WPcLix1.jpg', width=900)
    st.write("""Click on this link to gain access to the site: https://www.cbioportal.org/
            """)
    st.header('Data Analysis and Visualization')
    st.write("""By analyzing various factors such as age, gender, origin, prior medical history, genetic mutations, gene expression, gene methylation,
             and tumor characteristics, our models provide valuable insights into prognosis of patients affected by colorectal cancer.
             Click on the next link to gain access to data analysis and visualization: https://public.tableau.com/app/profile/juan.garc.a.moreno/viz/ColorectalcancerTCGAproject/CRCstory?publish=yes
            """)
      
def predictions():
    st.title('CRC Prognosis')
     # Add introductory text
    st.write('Please fill out the following form to get the patient prognosis.')
    # Define input fields
    features_patient_data  = {
            'prior_malignancy': 'Yes/No',
            'gender': 'Male/Female',
            'age_at_index': 'Numerical'
        }   
    features_mutational_status = {
            'apc': 'Yes/No',
            'tp53': 'Yes/No',
            'kras': 'Yes/No',
            'muc16': 'Yes/No',
            'pik3ca': 'Yes/No',
            'fat4': 'Yes/No',
            'lrp1b': 'Yes/No',
            'csmd3': 'Yes/No',
            'fat3': 'Yes/No',
            'fbxw7': 'Yes/No',
            'ptprt': 'Yes/No',
            'mll4': 'Yes/No',
            'nbea': 'Yes/No',
            'arid1a': 'Yes/No',
            'fam123b': 'Yes/No',
            'smad4': 'Yes/No',
            'atm': 'Yes/No',
            'tcf7l2': 'Yes/No',
            'braf': 'Yes/No',
            'zfhx3': 'Yes/No',
            'robo2': 'Yes/No',
            'rnf43': 'Yes/No',
            'rnf213': 'Yes/No',
            'grin2a': 'Yes/No',
            'fat1': 'Yes/No',
            'erbb4': 'Yes/No',
            'tnc': 'Yes/No',
            'mll3': 'Yes/No',
            'acvr2a': 'Yes/No',
            'trrap': 'Yes/No',
            'akap9': 'Yes/No',
            'crebbp': 'Yes/No',
            'cntnap2': 'Yes/No',
            'birc6': 'Yes/No',
            'prex2': 'Yes/No',
            'atrx': 'Yes/No',
            'ank1': 'Yes/No',
            'card11': 'Yes/No',
            'ctnna2': 'Yes/No',
            'fam47c': 'Yes/No',
            'cdh10': 'Yes/No',
            'bcl9l': 'Yes/No',
            'myh11': 'Yes/No',
            'brca2': 'Yes/No',
            'cdh4': 'Yes/No',
            'ros1': 'Yes/No',
            'dcc': 'Yes/No',
            'bcl9': 'Yes/No',
            'ptpn13': 'Yes/No',
            'bcorl1': 'Yes/No'
        }
    features_gene_expression = {
            'ACVR2A_exp': 'Numerical',
            'AKAP9_exp': 'Numerical',
            'ANK1_exp': 'Numerical',
            'APC_exp': 'Numerical',
            'ARID1A_exp': 'Numerical',
            'ATM_exp': 'Numerical',
            'ATRX_exp': 'Numerical',
            'BCL9_exp': 'Numerical',
            'BCL9L_exp': 'Numerical',
            'BCORL1_exp': 'Numerical',
            'BIRC6_exp': 'Numerical',
            'BRAF_exp': 'Numerical',
            'BRCA2_exp': 'Numerical',
            'CARD11_exp': 'Numerical',
            'CDH4_exp': 'Numerical',
            'CNTNAP2_exp': 'Numerical',
            'CREBBP_exp': 'Numerical',
            'CSMD3_exp': 'Numerical',
            'CTNNA2_exp': 'Numerical',
            'ERBB4_exp': 'Numerical',
            'FAM123B_exp': 'Numerical',
            'FAT1_exp': 'Numerical',
            'FAT3_exp': 'Numerical',
            'FAT4_exp': 'Numerical',
            'FBXW7_exp': 'Numerical',
            'GRIN2A_exp': 'Numerical',
            'KRAS_exp': 'Numerical',
            'LRP1B_exp': 'Numerical',
            'MLL3_exp': 'Numerical',
            'MLL4_exp': 'Numerical',
            'MUC16_exp': 'Numerical',
            'MYH11_exp': 'Numerical',
            'NBEA_exp': 'Numerical',
            'PIK3CA_exp': 'Numerical',
            'PREX2_exp': 'Numerical',
            'PTPN13_exp': 'Numerical',
            'PTPRT_exp': 'Numerical',
            'RNF213_exp': 'Numerical',
            'RNF43_exp': 'Numerical',
            'ROBO2_exp': 'Numerical',
            'ROS1_exp': 'Numerical',
            'SMAD4_exp': 'Numerical',
            'TCF7L2_exp': 'Numerical',
            'TNC_exp': 'Numerical',
            'TP53_exp': 'Numerical',
            'TRRAP_exp': 'Numerical',
            'ZFHX3_exp': 'Numerical'
        }
    features_gene_methylation = {
            'ACVR2A_met': 'Numerical',
            'AKAP9_met': 'Numerical',
            'ANK1_met': 'Numerical',
            'APC_met': 'Numerical',
            'ARID1A_met': 'Numerical',
            'ATRX_met': 'Numerical',
            'BCL9_met': 'Numerical',
            'BCL9L_met': 'Numerical',
            'BCORL1_met': 'Numerical',
            'BIRC6_met': 'Numerical',
            'BRAF_met': 'Numerical',
            'BRCA2_met': 'Numerical',
            'CARD11_met': 'Numerical',
            'CDH10_met': 'Numerical',
            'CDH4_met': 'Numerical',
            'CNTNAP2_met': 'Numerical',
            'CSMD3_met': 'Numerical',
            'CTNNA2_met': 'Numerical',
            'DCC_met': 'Numerical',
            'ERBB4_met': 'Numerical',
            'FAM123B_met': 'Numerical',
            'FAT1_met': 'Numerical',
            'FBXW7_met': 'Numerical',
            'GRIN2A_met': 'Numerical',
            'KRAS_met': 'Numerical',
            'LRP1B_met': 'Numerical',
            'MLL3_met': 'Numerical',
            'MLL4_met': 'Numerical',
            'MYH11_met': 'Numerical',
            'PIK3CA_met': 'Numerical',
            'PREX2_met': 'Numerical',
            'PTPN13_met': 'Numerical',
            'PTPRT_met': 'Numerical',
            'RNF213_met': 'Numerical',
            'RNF43_met': 'Numerical',
            'ROS1_met': 'Numerical',
            'SMAD4_met': 'Numerical',
            'TCF7L2_met': 'Numerical',
            'TNC_met': 'Numerical',
            'TRRAP_met': 'Numerical',
            'ZFHX3_met': 'Numerical'
        }
    features_pathologic_stage = {
            'ajcc_pathologic_stage_Stage I': 'Yes/No',
            'ajcc_pathologic_stage_Stage IA': 'Yes/No',
            'ajcc_pathologic_stage_Stage II': 'Yes/No',
            'ajcc_pathologic_stage_Stage IIA': 'Yes/No',
            'ajcc_pathologic_stage_Stage IIB': 'Yes/No',
            'ajcc_pathologic_stage_Stage IIC': 'Yes/No',
            'ajcc_pathologic_stage_Stage III': 'Yes/No',
            'ajcc_pathologic_stage_Stage IIIA': 'Yes/No',
            'ajcc_pathologic_stage_Stage IIIB': 'Yes/No',
            'ajcc_pathologic_stage_Stage IIIC': 'Yes/No',
            'ajcc_pathologic_stage_Stage IV': 'Yes/No',
            'ajcc_pathologic_stage_Stage IVA': 'Yes/No',
            'ajcc_pathologic_stage_Stage IVB': 'Yes/No'
        }
    features_tissue_or_organ_of_origin = {
            'tissue_or_organ_of_origin_Ascending colon': 'Yes/No',
            'tissue_or_organ_of_origin_Cecum': 'Yes/No',
            'tissue_or_organ_of_origin_Colon': 'Yes/No',
            'tissue_or_organ_of_origin_Connective': 'Yes/No',
            'tissue_or_organ_of_origin_Descending colon': 'Yes/No',
            'tissue_or_organ_of_origin_Hepatic flexure of colon': 'Yes/No',
            'tissue_or_organ_of_origin_Rectosigmoid junction': 'Yes/No',
            'tissue_or_organ_of_origin_Rectum': 'Yes/No',
            'tissue_or_organ_of_origin_Sigmoid colon': 'Yes/No',
            'tissue_or_organ_of_origin_Splenic flexure of colon': 'Yes/No',
            'tissue_or_organ_of_origin_Transverse colon': 'Yes/No',
            'tissue_or_organ_of_origin_Unknown primary site': 'Yes/No'
        }
    features_primary_diagnosis = {
            'primary_diagnosis_Adenocarcinoma': 'Yes/No',
            'primary_diagnosis_Adenocarcinoma in tubolovillous adenoma': 'Yes/No',
            'primary_diagnosis_Adenocarcinoma with mixed subtypes': 'Yes/No',
            'primary_diagnosis_Adenocarcinoma with neuroendocrine differentiation': 'Yes/No',
            'primary_diagnosis_Adenosquamous carcinoma': 'Yes/No',
            'primary_diagnosis_Carcinoma': 'Yes/No',
            'primary_diagnosis_Mucinous adenocarcinoma': 'Yes/No',
            'primary_diagnosis_Papillary adenocarcinoma': 'Yes/No',
            'primary_diagnosis_Tubular adenocarcinoma': 'Yes/No'
        }
    features_morphology = {
            'morphology_8010/3': 'Yes/No',
            'morphology_8140/3': 'Yes/No',
            'morphology_8211/3': 'Yes/No',
            'morphology_8255/3': 'Yes/No',
            'morphology_8260/3': 'Yes/No',
            'morphology_8263/3': 'Yes/No',
            'morphology_8480/3': 'Yes/No',
            'morphology_8560/3': 'Yes/No',
            'morphology_8574/3': 'Yes/No'
        }
    features_TNM = {
            'ajcc_pathologic_t_T1': 'Yes/No',
            'ajcc_pathologic_t_T2': 'Yes/No',
            'ajcc_pathologic_t_T3': 'Yes/No',
            'ajcc_pathologic_t_T4': 'Yes/No',
            'ajcc_pathologic_t_T4a': 'Yes/No',
            'ajcc_pathologic_t_T4b': 'Yes/No',
            'ajcc_pathologic_t_Tis': 'Yes/No',
            'ajcc_pathologic_n_N0': 'Yes/No',
            'ajcc_pathologic_n_N1': 'Yes/No',
            'ajcc_pathologic_n_N1a': 'Yes/No',
            'ajcc_pathologic_n_N1b': 'Yes/No',
            'ajcc_pathologic_n_N1c': 'Yes/No',
            'ajcc_pathologic_n_N2': 'Yes/No',
            'ajcc_pathologic_n_N2a': 'Yes/No',
            'ajcc_pathologic_n_N2b': 'Yes/No',
            'ajcc_pathologic_m_M0': 'Yes/No',
            'ajcc_pathologic_m_M1': 'Yes/No',
            'ajcc_pathologic_m_M1a': 'Yes/No',
            'ajcc_pathologic_m_M1b': 'Yes/No'
        }

    # Add input fields for user input - Patient Data
    st.header('Patient Data')
    user_input_patient_data = {}
    for feature, input_type in features_patient_data.items():
        if input_type == 'Yes/No':
            user_input_patient_data[feature] = st.selectbox(feature, ['No', 'Yes'])  # Display Yes/No dropdown
        elif input_type == 'Male/Female':
            user_input_patient_data[feature] = st.selectbox(feature, ['Male', 'Female'])  # Display Male/Female dropdown
        elif input_type == 'Numerical':
            if feature == 'age_at_index':
                user_input_patient_data[feature] = st.number_input('age', min_value=0, max_value=120, value=50)  # Age input
            else:
                user_input_patient_data[feature] = st.number_input(feature, value=0)  # Other numerical inputs
    
    # Add input fields for user input - Mutational Status
    st.header('Mutational Status')
    user_input_mutational_status = {}
    for feature, input_type in features_mutational_status.items():
        if input_type == 'Yes/No':
            user_input_mutational_status[feature] = st.selectbox(feature, ['No', 'Yes'])  # Display Yes/No dropdown
        # Add handling for other input types

    # Add input fields for user input - Expression
    st.header('Gene expression')
    user_input_gene_expression = {}
    for feature, input_type in features_gene_expression.items():
        if input_type == 'Numerical':
            user_input_gene_expression[feature] = st.number_input(feature, value=0)  # Numerical inputs
    
    # Add input fields for user input - Methylation
    st.header('Gene methylation')
    st.write("""Please, enter a numerical value between 0 and 1
             """)
    user_input_gene_methylation = {}
    for feature, input_type in features_gene_methylation.items():
        if input_type == 'Numerical':
            user_input_gene_methylation[feature] = st.number_input(feature, min_value=0, max_value=1, value=0)  # Numerical inputs
    
    # Add input fields for user input - pathologic stage
    st.header('Pathologic stage')
    user_input_pathologic_stage = {}
    for feature, input_type in features_pathologic_stage.items():
        if input_type == 'Yes/No':
            user_input_pathologic_stage[feature] = st.selectbox(feature, ['No', 'Yes'])  # Display Yes/No dropdown
    
    # Add input fields for user input - tissue or organ of origin
    st.header('Tissue or organ of origin')
    user_input_organ_of_origin = {}
    for feature, input_type in features_tissue_or_organ_of_origin.items():
        if input_type == 'Yes/No':
            user_input_organ_of_origin[feature] = st.selectbox(feature, ['No', 'Yes'])  # Display Yes/No dropdown

    # Add input fields for user input - primary diagnosis
    st.header('Primary diagnosis')
    user_input_primary_diagnosis = {}
    for feature, input_type in features_primary_diagnosis.items():
        if input_type == 'Yes/No':
            user_input_primary_diagnosis[feature] = st.selectbox(feature, ['No', 'Yes'])  # Display Yes/No dropdown
    
    # Add input fields for user input - morphology
    st.header('Tumor morphology')
    user_input_morphology = {}
    for feature, input_type in features_morphology.items():
        if input_type == 'Yes/No':
            user_input_morphology[feature] = st.selectbox(feature, ['No', 'Yes'])  # Display Yes/No dropdown
    
    # Add input fields for user input - TNM
    st.header('TNM staging')
    user_input_TNM = {}
    for feature, input_type in features_TNM.items():
        if input_type == 'Yes/No':
            user_input_TNM[feature] = st.selectbox(feature, ['No', 'Yes'])  # Display Yes/No dropdown
    
    # Combine user inputs from different categories
    user_input = {**user_input_patient_data, **user_input_mutational_status, **user_input_gene_expression, 
                  **user_input_gene_methylation, **user_input_pathologic_stage, **user_input_organ_of_origin,
                  **user_input_primary_diagnosis,**user_input_morphology,**user_input_TNM}
    
    # Convert user inputs to model input format
    for feature, value in user_input.items():
        if input_type  == 'Yes/No':
            user_input[feature] = 1 if value == 'Yes' else 0  # Convert Yes/No to 1/0
        elif input_type  == 'Male/Female':
            user_input[feature] = 1 if value == 'Male' else 0  # Convert Male/Female to 1/0
    
    input_df = pd.DataFrame([user_input])

    if st.button('Predict'):
        # Make predictions using each model
        for model_name, model in models.items():
            with st.expander(model_name):
                prediction = get_prediction(model, input_df)
                prediction_text = 'Poor Prognosis' if prediction[0] == 1 else 'Good Prognosis'
                st.write(f'Prediction: {prediction_text}')

                # Calculate accuracy for this specific prediction
                accuracy = None
                rmse = None
                if validation_data is not None:
                    y_true = validation_data['vital_status'] 
                    y_pred = model.predict(validation_data.drop('vital_status', axis=1))
                    
                    # Calculate accuracy
                    accuracy = accuracy_score(y_true, y_pred)
                    st.write(f'Accuracy: {accuracy}')

                    # Calculate RMSE
                    rmse = mean_squared_error(y_true, y_pred, squared=False)
                    st.write(f'RMSE: {rmse}')


def main():
    pages = {
        "Home": home,
        "About Colorectal Cancer": about_colorectal_cancer,
        "Data source and visualization": data_source,
        "Predictions": predictions
    }
    st.markdown("""
        <style>
            .sidebar-image-container {
                display: flex;
                justify-content: center;
                margin-bottom: 20px; /* Adjust as needed */
            }
        </style>
    """, unsafe_allow_html=True)

    # Add the image to the sidebar
    st.sidebar.markdown('<div class="sidebar-image-container"><img src="https://imgur.com/ivXCEUY.jpg" width="300"/></div>', unsafe_allow_html=True)
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    page = pages[selection]
    page()

if __name__ == '__main__':
    main()
    