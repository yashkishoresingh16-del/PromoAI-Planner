# Deploy PromoAI to Streamlit Community Cloud

Extract this package. Put the contents of the extracted folder in the root of a
GitHub repository, preserving the data/ and dist/ folders. The entry point is
streamlit_app.py. app.py supplies the live solver interface.

At https://share.streamlit.io choose Create app and deploy from your GitHub
repository. Choose its main branch and streamlit_app.py as the file path.
Under Advanced settings select Python 3.12. No app secrets are required.
After deployment, open the URL in an incognito window to check viewer access.

For a local test on Windows:
    py -m pip install -r requirements.txt
    py -m streamlit run streamlit_app.py

The app starts with verified synthetic results. Expand the live optimization
section and submit the default caps with the upload boxes blank to run
LightGBM and OR-Tools SCIP on the included sample. For a public classroom
deployment use synthetic data. Control repository and app access before
uploading confidential business information.
