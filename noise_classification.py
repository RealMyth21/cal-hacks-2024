import os
from msclap import CLAP
import torch.nn.functional as F

# Define classes for zero-shot
classes = ["gun shot", "fire", "screaming", "footsteps", "static"]
ground_truth = ["static"]  # Example ground truth, can be adjusted based on actual data
# Add prompt
prompt = "this is a sound of "
class_prompts = [prompt + x for x in classes]

# Relative path to your audio file
relative_path = "call_audios/call_108.mp3"

# Get the script's directory path
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the audio file
audio_files = [os.path.join(script_dir, relative_path)]

# Load and initialize CLAP
clap_model = CLAP(version="2023", use_cuda=False)  # Set use_cuda=True if using GPU

# Compute text embeddings from natural text
text_embeddings = clap_model.get_text_embeddings(class_prompts)

# Compute the audio embeddings from the audio file
audio_embeddings = clap_model.get_audio_embeddings(audio_files, resample=True)

# Compute the similarity between audio_embeddings and text_embeddings
similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)

# Apply softmax to get probabilities
similarity = F.softmax(similarity, dim=1)

# Get the top predictions
values, indices = similarity[0].topk(5)

# Print the results
print("Ground Truth: {}".format(ground_truth))
print("Top predictions:\n")
for value, index in zip(values, indices):
    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
