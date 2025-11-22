<h1>🎓 AI-Powered Student Success Prediction System</h1>


An intelligent machine learning system designed to predict student academic success based on historical performance, demographics, and study behaviors. Built with XGBoost for high-performance classification and deployed as an interactive web application using Streamlit.


<h3>✨ Key Features</h3>


📌 AI-Powered Academic Forecasting

Gradient Boosting Model: Utilizes XGBoost (Extreme Gradient Boosting) to classify students into "Pass" or "Fail" categories with high precision. Mid-Term Prediction: Leverages interim grades (G1, G2) to forecast final outcomes before the academic year ends. Early Warning System: Identifies students at risk of failure to facilitate timely interventions.

📌 Detailed Feature Analysis


Key Drivers Visualization: Automatically identifies and plots the top factors influencing student performance (e.g., prior grades, absences, alcohol consumption). Holistic Assessment: Considers academic, social, and demographic factors. Interpretable Insights: Provides clear reasons why a prediction was made.

📌 Interactive Web Dashboard


User-Friendly Interface: Intuitive sidebar for inputting student data. Real-Time Scoring: Instant calculation of success probability. Dynamic Visualization: "Balloons" for success and "Warning" alerts for at-risk cases. Personalized Reporting: Generates a summary specific to the student's name and class.

📌 Transparent Results


Probability Score: Percentage likelihood of passing the final exam (0-100%). Actionable Feedback: Suggests when intervention is recommended based on low confidence or failure predictions.


<h3>🛠 Tech Stack</h3>


<table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: left;"> <thead> <tr style="background-color: #f2f2f2;"> <th>Component</th> <th>Technology</th> </tr> </thead> <tbody> <tr> <td>Backend</td> <td>Python 3.10</td> </tr> <tr> <td>Machine Learning</td> <td>XGBoost (Extreme Gradient Boosting)</td> </tr> <tr> <td>Web Framework</td> <td>Streamlit</td> </tr> <tr> <td>Data Processing</td> <td>pandas, NumPy</td> </tr> <tr> <td>Visualization</td> <td>Matplotlib, Seaborn</td> </tr> <tr> <td>Model Training</td> <td>scikit-learn (GridSearch, StandardScaler)</td> </tr> </tbody> </table>

<h2>📊 Dataset</h2> <h3>Source</h3> UCI Machine Learning Repository: Student Performance Data Set (Cortez and Silva, 2008)


<h3>Dataset Details</h3> Contains two CSV files derived from Portuguese secondary schools:


student-mat.csv - Performance in Mathematics


student-por.csv - Performance in Portuguese Language


<h3>Dataset Statistics</h3>

Total Records: ~1,044 student records (merged) Target Variable: G3 (Final Grade) >= 10 (Pass) vs < 10 (Fail) Features: 30+ raw features covering social, gender, and study data


<h3>Features Used</h3> Academic Features (Numeric)


G1 (First Period Grade: 0-20), G2 (Second Period Grade: 0-20), Absences (0-93), Failures (Past class failures), Study Time (Weekly hours), Travel Time.

Demographic & Social Features (Categorical/Binary)

Personal: Age, Gender, Health Status. Family: Parents' Education (Medu, Fedu), Parents' Jobs (Mjob, Fjob), Family Support (famsup). Lifestyle: Alcohol Consumption (Dalc, Walc), Going Out (goout), Internet Access, Romantic Relationships, Extra-curricular Activities.

Total Input Features: ~45 (after One-Hot Encoding)


<h3>Data Preprocessing</h3>

Merging: Combined Math and Portuguese datasets for a larger training corpus. Binary Mapping: Converted 'yes/no' columns (e.g., paid, internet) to 0/1. One-Hot Encoding: Applied to nominal variables like Mjob (Mother's Job) and reason. Target Engineering: Created a binary success variable where G3 ≥ 10 is 1 (Pass) and G3 < 10 is 0 (Fail). Scaling: Applied Standard Scaling to numerical features to normalize ranges.

<h3>🎛 Model Architecture & Training</h3>


<h2>Model Configuration</h2>


<table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: left;"> <thead> <tr style="background-color: #f2f2f2;"> <th>Parameter</th> <th>Value</th> <th>Description</th> </tr> </thead> <tbody> <tr> <td>Algorithm</td> <td>XGBClassifier</td> <td>Tree-based gradient boosting ensemble</td> </tr> <tr> <td>Objective</td> <td>binary:logistic</td> <td>Logistic regression for binary classification</td> </tr> <tr> <td>Tree Method</td> <td>hist</td> <td>Histogram-based split finding (Optimized for speed)</td> </tr> <tr> <td>N_Estimators</td> <td>100</td> <td>Number of gradient boosted trees</td> </tr> <tr> <td>Max Depth</td> <td>5</td> <td>Maximum depth of a tree (controls overfitting)</td> </tr> <tr> <td>Learning Rate</td> <td>0.1</td> <td>Step size shrinkage used in update to prevent overfitting</td> </tr> <tr> <td>Evaluation Metric</td> <td>logloss</td> <td>Logarithmic Loss</td> </tr> </tbody> </table>


<h3>Training Process</h3>

<h2>Data Split:</h2>


Training: 80% Testing: 20%


<h2>Stratification:</h2>

Applied stratified sampling to ensure the proportion of passing and failing students remains consistent between training and testing sets.


<h2>Feature Engineering Strategy:</h2>

Leakage Prevention: Dropped G3 (Final Grade) from inputs to prevent the model from "cheating." Context Awareness: Included subject column to distinguish between Math and Portuguese contexts.


<h3>📊 Model Performance Metrics</h3>


<table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: left;"> <thead> <tr style="background-color: #f2f2f2;"> <th>Metric</th> <th>Score</th> <th>Description</th> </tr> </thead> <tbody> <tr> <td>Accuracy</td> <td>90.91%</td> <td>Percentage of correct predictions on the test set</td> </tr> <tr> <td>Precision</td> <td>~92%</td> <td>Reliability of "Pass" predictions</td> </tr> <tr> <td>Recall</td> <td>~94%</td> <td>Ability to detect actual passing students</td> </tr> <tr> <td>F1-Score</td> <td>~93%</td> <td>Harmonic mean of precision and recall</td> </tr> <tr> <td>ROC-AUC</td> <td>0.94</td> <td>Excellent capability to distinguish between Pass/Fail classes</td> </tr> </tbody> </table>


<h2>Strengths</h2>


✅ High Accuracy: ~91% accuracy makes it reliable for administrative use. ✅ Speed: XGBoost with histogram methods trains and predicts in milliseconds. ✅ Robustness: Handles both categorical (Jobs) and numerical (Grades) data seamlessly. ✅ Interpretability: Feature importance analysis clearly highlights G2 and Absences as primary predictors.


<h2>Model Limitations</h2> ⚠ Grade Dependency: Heavily relies on G1/G2 grades; prediction accuracy drops if these are unavailable (start of year). ⚠ Demographic Bias: Potential biases in historical data regarding gender or urban/rural status. ⚠ Static Nature: Does not account for sudden life events occurring after data collection.


<h3>🧪 Example Predictions</h3>


<h2>The "High Achiever"</h2>


Input:

G1 Grade: 16

G2 Grade: 17

Absences: 2

Study Time: >10 hrs

Alcohol Consumption: Low

Output: → Success Probability: 98.4% → Prediction: ✅ PASS → Insight: Strong past performance and low absences drive this result.


<h2>The "At-Risk" Student</h2> Input:

G1 Grade: 8

G2 Grade: 7

Absences: 15

Study Time: <2 hrs

Alcohol Consumption: High

Output: → Success Probability: 12.5% → Prediction: ❌ FAIL → Insight: Warning triggered. Recommended for immediate counseling.


<h2>The "Borderline" Case</h2>


Input:

G1 Grade: 9

G2 Grade: 10

Absences: 6

Study Time: 2-5 hrs

Family Support: Yes

Output: → Success Probability: 55.3% → Prediction: ✅ PASS (Low Confidence) → Insight: Student is on the edge; family support pushed the probability slightly over 50%.


<h3>🚀 Future Enhancements</h3> <h2>Research Directions</h2>


Behavioral Tracking: Integrate real-time LMS login data. NLP Integration: Analyze teacher notes and feedback comments using Sentiment Analysis. Recommender System: Suggest specific study materials for students identified as "At-Risk." Start-of-Year Model: Train a separate model excluding G1/G2 for Day 1 predictions. Bias Audit: Rigorous testing to ensure fairness across gender and socioeconomic groups.
