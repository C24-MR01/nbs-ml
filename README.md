# nbs-ml

***How to use deployed app_synopsys.py***

1. Checkout to branch "main"
2. type command "pip install-r requirements.txt" ( make sure you're in root folder of nbs-ml)

   ```
   pip install -r requirements.txt
   ```
3. route to folder deploy-google, type below code in your terminal/cmd

   ```
   cd deploy-google/
   ```
4. Set the flask app  to initialize the app, type below code in your terminal/cmd

   ```
   # For windows or mac:
   set FLASK_APP=app.py
   ```
5. run the website by your terminal/cmd. type below code in your terminal/cmd

   ```
   python app.py
   ```
6. open your browser and enter "http://127.0.0.1:5000/recommend-synopsys?id=100". **The id value is replaceable.**
7. You can try other endpoint "http://127.0.0.1:5000/recommend-collab?id=2", "http://127.0.0.1:5000/recommend-metadata?id=2"
