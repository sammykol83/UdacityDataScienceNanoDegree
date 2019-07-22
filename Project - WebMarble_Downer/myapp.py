from myapp import app

deploy = False  # Change to True if you want to deploy via Heroku

if deploy:
    app.run(host='0.0.0.0', port=3001, debug=True)
else:
    app.run(host='127.0.0.1', port=5000, debug=True)
