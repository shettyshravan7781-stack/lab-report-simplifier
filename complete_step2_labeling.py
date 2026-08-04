import os
import pandas as pd
import numpy as np

def run_complete_step2():
    print("==============================================================")
    print("   Finalizing Step 2: Adding Natural Language Interpretations ")
    print("==============================================================\n")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    dataset_path = os.path.join(OUTPUT_DIR, "ML_Ready_Hospital_Dataset.csv")
    
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(BASE_DIR, "ML_Ready_Hospital_Dataset.csv")
        
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path, low_memory=False)

    # -----------------------------------------------------------------
    # TASK B: NATURAL LANGUAGE INTERPRETATION GENERATOR
    # -----------------------------------------------------------------
    print("\nExecuting Task B: Generating Patient-Friendly Explanations...")

    def generate_patient_explanation(row):
        explanations = []
        
        # 1. Diabetes Explanation
        d_risk = row.get('Target_Diabetes_Risk', 0)
        if d_risk == 2:
            explanations.append("High Glucose/HbA1c levels indicate Diabetic range. Endocrinologist consultation recommended.")
        elif d_risk == 1:
            explanations.append("Slightly elevated blood glucose levels (Prediabetes range). Dietary modification advised.")
        else:
            explanations.append("Blood sugar levels are within normal range.")
            
        # 2. Kidney Explanation
        k_risk = row.get('Target_Kidney_Risk', 0)
        if k_risk == 2:
            explanations.append("Elevated Creatinine or reduced eGFR indicates significant kidney filtration stress.")
        elif k_risk == 1:
            explanations.append("Mild variation in kidney parameters. Hydration and follow-up monitoring advised.")
            
        # 3. Anemia Explanation
        a_risk = row.get('Target_Anemia_Risk', 0)
        if a_risk == 2:
            explanations.append("Hemoglobin and RBC count are significantly low (Moderate/Severe Anemia). Iron intake or hematology review needed.")
        elif a_risk == 1:
            explanations.append("Hemoglobin is slightly below optimal threshold (Mild Anemia).")

        # 4. Liver Explanation
        l_risk = row.get('Target_Liver_Risk', 0)
        if l_risk == 1:
            explanations.append("Elevated liver enzymes (ALT/AST or Bilirubin). Consider reviewing liver health and dietary habits.")
            
        # 5. Lipid Explanation
        lp_risk = row.get('Target_Lipid_Risk', 0)
        if lp_risk == 2:
            explanations.append("Lipid profile shows high cholesterol/triglycerides indicating elevated cardiovascular risk.")
        elif lp_risk == 1:
            explanations.append("Borderline elevated lipid levels. Exercise and dietary adjustments recommended.")
            
        return " | ".join(explanations)

    df['Patient_Explanation_Summary'] = df.apply(generate_patient_explanation, axis=1)

    # Save finalized dataset
    df.to_csv(dataset_path, index=False)
    print(f"✓ Successfully added 'Patient_Explanation_Summary' to '{dataset_path}'")
    print(f"Final Dataset Shape: {df.shape}")
    
    print("\n==============================================================")
    print("   STEP 2 IS NOW 100% COMPLETE (Task A + Task B Finished!)    ")
    print("==============================================================")

if __name__ == "__main__":
    run_complete_step2()