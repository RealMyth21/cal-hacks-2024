import eventlet

eventlet.monkey_patch()

import os
import json
import asyncio
from flask import Flask, request, url_for, redirect, render_template
from flask_socketio import SocketIO, emit
import websockets
from hume import HumeVoiceClient
from dotenv import load_dotenv
import base64

load_dotenv()

app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet")

HUME_WS_URI = "wss://api.hume.ai/v0/stream/models"
HUME_API_KEY = os.getenv("HUME_API_KEY")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


app.config["UPLOAD_FOLDER"] = "static\css/audios"
UPLOAD_FOLDER = "static\css/audios"


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    formats = {"mp3", "wav", "webm"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in formats


app.config["DEBUG"] = True


@app.route("/demo", methods=["GET", "POST"])
def demo():
    try:
        if request.method == "POST":
            # Check if the post request has the file part
            if "file" not in request.files:
                print("No file part in request")
                return redirect(request.url)
            file = request.files["file"]
            # If the user does not select a file, the browser submits an empty file without a filename
            if file.filename == "":
                print("No selected file")
                return redirect(request.url)
            """
            if file:
                filename = file.filename
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                print(f"File saved: {filename}")
                return redirect(url_for('demo'))
            """
            if file and allowed_file(file.filename):
                filename = file.filename
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
                print(f"File saved: {filename}")
                return redirect(url_for("demo"))

        return render_template("demo.html")
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred during the file upload process."


async def connect_to_hume(websocket, audio_chunks):
    try:
        async with websockets.connect(HUME_WS_URI) as ws:
            await ws.send(json.dumps({"api_key": HUME_API_KEY}))

            while True:
                chunk = await audio_chunks.get()
                if chunk is None:
                    break
                await ws.send(json.dumps({"type": "audio", "content": chunk}))

                response = await ws.recv()
                result = json.loads(response)
                await websocket.send(json.dumps(result))
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")


@app.route("/analyze")
def analyze():
    return render_template("analyze.html")


@socketio.on("start_analysis")
def handle_start_analysis():
    audio_chunks = asyncio.Queue()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    websocket = loop.run_until_complete(
        websockets.serve(
            lambda ws, path: connect_to_hume(ws, audio_chunks), "localhost", 8765
        )
    )

    def audio_handler(data):
        loop.call_soon_threadsafe(audio_chunks.put_nowait, data)

    socketio.on("audio_chunk", audio_handler)

    socketio.start_background_task(websocket)
    emit("status", {"status": "Started real-time analysis"})


@socketio.on("stop_analysis")
def handle_stop_analysis():
    emit("status", {"status": "Stopped real-time analysis"})


async def send_to_hume(encoded_data):
    try:
        async with websockets.connect(HUME_WS_URI) as ws:
            await ws.send(json.dumps({"api_key": HUME_API_KEY, "data": encoded_data}))

            response = await ws.recv()
            result = json.loads(response)
            print("Received from Hume:", result)
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")


def encode_data(filepath: str) -> str:
    with open(filepath, "rb") as fp:
        bytes_data = fp.read()
        encoded_data = base64.b64encode(bytes_data).decode("utf-8")
    return encoded_data


def allowed_file(filename):
    formats = {"mp3", "wav", "webm"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in formats


@socketio.on("connect")
def test_connect():
    print("Client connected")


@socketio.on("disconnect")
def test_disconnect():
    print("Client disconnected")


if __name__ == "__main__":
    socketio.run(app, debug=True)
