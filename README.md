Cyber Incident Response and Reporting Platform

A comprehensive web application designed to help individuals and communities understand, analyze, and report both cyber and physical security incidents. The platform combines an intuitive interface, a machine learning classifier, educational resources, and real-time analytics to provide an end-to-end incident management experience.

Features
Incident Analysis

Incident Input: Users can submit incident descriptions for automated assessment.

ML-Powered Classification: A trained machine learning model predicts the likely cyberattack type.

Guided Recommendations: Based on predictions, the system provides actionable next steps.

Incident Reporting

Detailed Reporting Form: Supports both anonymous and detailed incident submissions.

Structured Information Capture: Collects location, dates, descriptions, and other relevant fields.

Cyber Awareness & Education

Learning Hub: Includes resources on types of crimes, reporting procedures, and user rights.

Educational Media: Features a carousel of awareness videos for improved understanding.

Real-Time Analytics

Interactive Charts: Visual representations of cybercrime trends and statistics.

Geographical Mapping: Displays regional distribution of incidents using interactive maps.

Post-Attack Support

Mitigation Steps: Offers recommended actions to reduce further impact.

Helpline Directory: Provides contact details for official support organizations.

Team & Project Information

About Page: Showcases the vision, mission, and team members.

Contact Page: Allows users to request additional support or information.

Tech Stack
Frontend

HTML, CSS, Bootstrap for responsive UI.

JavaScript with Chart.js and Leaflet.js for charts and maps.

Backend

Python as the core development language.

Flask for the backend framework.

Machine Learning & Data Processing

scikit-learn for building the classification model.

pandas for data manipulation.

joblib for model persistence.

Visualization

Chart.js for interactive data charts.

Leaflet.js for interactive mapping.

Setup Instructions

Navigate to your project directory:

cd <your_project_directory>


Clone the repository:

git clone https://github.com/Maheshannayboeina/Cyber-Incident-Response.git


Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows


Install dependencies:

pip install -r requirements.txt


Run the application:

python app.py


Open your browser and navigate to:

http://127.0.0.1:5000

Usage

Incident Analysis: Enter a description on the home page to receive attack predictions and recommended actions.

Incident Reporting: Submit new reports through the “Report” page.

Learning Materials: Browse the “Learn” page for safety tips, crime information, and awareness videos.

Analytics: Review cybercrime statistics and map visualizations on the “Analytics” page.

Note: The analytics module currently uses sample data. To integrate real data, update the relevant data-fetching logic in app.py.

Contributing

Contributions are welcome. To propose enhancements or report issues, please open an issue or submit a pull request. Ensure that all code changes are tested before submission.

Disclaimer

This platform provides information and recommendations based on available data and machine learning predictions. It is not a substitute for professional cybersecurity services or legal consultation. Always seek qualified assistance for sensitive or high-risk incidents.

License

License information will be added soon.
