```bash
gcloud projects list
gcloud config set project ca-gcp-data-science-dev
gcloud auth application-default set-quota-project ca-gcp-data-science-dev
gcloud auth application-default login --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/spreadsheets,https://www.googleapis.com/auth/drive.file
```