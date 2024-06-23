import asyncio
import json

from hume import HumeStreamClient
from hume.models.config import ProsodyConfig

async def main():
    client = HumeStreamClient("IawcufxKtWqusn6UgKTvkOhIu7mkMv71VS1KEMmzCF97UKok")
    config = ProsodyConfig()
    async with client.connect([config]) as socket:
        result = await socket.send_file("HELP.mp3")
        formatted_result = format_result(result)
        print(formatted_result)
        with open("output.txt", "w") as outfile:
            outfile.write(formatted_result)

def format_result(result):
    formatted_result = ""
    for prediction in result["prosody"]["predictions"]:
        formatted_result += f"Time: {prediction['time']['begin']} - {prediction['time']['end']}\n"
        for emotion in prediction["emotions"]:
            formatted_result += f"{emotion['name']}: {emotion['score']}\n"
        formatted_result += "\n"
    return formatted_result

asyncio.run(main())

# Read the result file
with open("output.txt", "r") as file:
    lines = file.readlines()

# Parse the emotions and scores
emotions = {}
current_emotion = None
for line in lines:
    if line.startswith("Time"):
        continue
    if not line.strip():
        continue
    if ":" in line:
        emotion, score = line.strip().split(": ")
        emotions[emotion] = float(score)

# Sort the emotions by score in descending order
sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

# Write the sorted emotions to a new file
with open("sorted_output.txt", "w") as file:
    for emotion, score in sorted_emotions:
        file.write(f"{emotion}: {score}\n")
