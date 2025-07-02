to run the program 
in the root of project: python run_web.py
to update indexed vectors: in seperate terminal: python -m app.update_index
then Give signal using endpoint: curl -X POST http://127.0.0.1:5000/api/reload_pipeline 
