from app.main import create_app

app = create_app()

if __name__ == "__main__":
    # You can set host='0.0.0.0' for LAN testing
    app.run(host="127.0.0.1", port=5000, debug=True)
