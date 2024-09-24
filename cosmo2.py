import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from tkinter import scrolledtext, filedialog
from transformers import pipeline
from collections import deque
from PIL import Image
import clip
import sys
import threading
import signal
import warnings
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific TensorFlow warnings if not using GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
# Optionally, suppress FutureWarnings from transformers
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Configuration and Global Variables
emotion_to_vector = {
    "joy": [1, 0, 0],
    "sadness": [0, 1, 0],
    "anger": [0, 0, 1],
    "fear": [0, 0, 1],
    "surprise": [1, 0, 0],
    "disgust": [0, 0, 1],
    "neutral": [0, 1, 0],
}

# Initialize conversation memory as a global variable
conversation_memory = {}

# Initialization Functions
def initialize_emotion_detector():
    try:
        # Explicitly set 'clean_up_tokenization_spaces' if possible
        return pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            tokenizer="j-hartmann/emotion-english-distilroberta-base",
            # If the pipeline supports it, set 'clean_up_tokenization_spaces'
            # Note: Not all pipelines accept this parameter directly
        )
    except Exception as e:
        logger.error(f"Error loading the emotion detection model: {e}")
        sys.exit(1)

def initialize_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model, preprocess, device
    except Exception as e:
        logger.error(f"Error loading CLIP model: {e}")
        sys.exit(1)

emotion_detector = initialize_emotion_detector()

# Hebbian Learning Rule and Global Workspace Layer
def hebbian_update(weights, pre_activations, post_activations, hebbian_learning_rate):
    delta_w = hebbian_learning_rate * np.outer(pre_activations, post_activations)
    return weights + delta_w

class GlobalWorkspaceLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GlobalWorkspaceLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        return self.output(x)

class MetaLearningDQNModel:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.hebbian_learning_rate = 0.01
        self.model = self.build_model()
        self.gwl_model = GlobalWorkspaceLayer(state_size * 2, 128)

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state, emotion_vector, knowledge_context_vector, image_embedding):
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(self.action_size)
            logger.info(f"Random action chosen: {action}")
            return action
        # Concatenate all state representations
        state_input = np.concatenate([
            state,
            emotion_vector,
            knowledge_context_vector,
            image_embedding.flatten() if image_embedding is not None else np.zeros(128)
        ])
        state_input = np.reshape(state_input, [1, self.state_size * 2 + 128])
        q_values = self.model.predict(state_input)
        action = np.argmax(q_values[0])
        logger.info(f"Predicted action: {action} with Q-values: {q_values}")
        return action

# GUI Application
class ChatBotApp(tk.Tk):
    def __init__(self, retrain=False):
        super().__init__()
        self.title("Emotionally Adaptive Chatbot")
        # Initialize CLIP model
        self.clip_model, self.clip_preprocess, self.device = initialize_clip_model()
        # Initialize LSTM model with optional retraining
        self.lstm_model, self.tokenizer, self.max_len = train_lstm_model(
            'Dataset_2.csv', 
            'chatbot_checkpoint.h5',
            tokenizer_path='tokenizer.pkl',
            retrain=retrain
        )
        # Initialize DQN agent
        self.dqn_agent = MetaLearningDQNModel(state_size=3, action_size=3)
        self.setup_widgets()
        self.image_memory = {}

    def setup_widgets(self):
        self.text_area = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=60, height=20, font=("Arial", 14))
        self.text_area.grid(column=0, row=0, padx=10, pady=10)
        self.text_area.config(state=tk.DISABLED)

        self.user_input = tk.Entry(self, width=60, font=("Arial", 14))
        self.user_input.grid(column=0, row=1, padx=10, pady=10)

        self.send_button = tk.Button(self, text="Send", command=self.send_message, width=10, font=("Arial", 14))
        self.send_button.grid(column=0, row=2, padx=10, pady=10, sticky="e")

        self.image_button = tk.Button(self, text="Send Image", command=self.upload_image, width=10, font=("Arial", 14))
        self.image_button.grid(column=0, row=2, padx=10, pady=10, sticky="w")
        
        # Add a Retrain Button
        self.retrain_button = tk.Button(self, text="Retrain Model", command=self.retrain_model, width=15, font=("Arial", 14))
        self.retrain_button.grid(column=0, row=3, padx=10, pady=10, sticky="w")

    def send_message(self):
        user_text = self.user_input.get()
        if user_text:
            self.display_message("You: " + user_text)
            threading.Thread(target=self.process_and_respond, args=(1, user_text)).start()
        self.user_input.delete(0, tk.END)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.display_message("Image uploaded: " + os.path.basename(file_path))
            threading.Thread(target=self.process_image_and_respond, args=(1, file_path)).start()
    
    def retrain_model(self):
        # Path to the new dataset
        new_dataset_path = filedialog.askopenfilename(title="Select New Dataset", filetypes=[("CSV files", "*.csv")])
        if new_dataset_path:
            self.display_message("Retraining the model with the new dataset...")
            threading.Thread(target=self._retrain_model_thread, args=(new_dataset_path,)).start()

    def _retrain_model_thread(self, new_dataset_path):
        # Update the model with new data
        self.lstm_model, self.tokenizer, self.max_len = train_lstm_model(
            new_dataset_path, 
            'chatbot_checkpoint.h5',
            tokenizer_path='tokenizer.pkl',
            retrain=True
        )
        self.display_message("Model retraining completed.")

    def display_message(self, message):
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, f"\n{message}")
        self.text_area.config(state=tk.DISABLED)
        self.text_area.see(tk.END)

    def process_and_respond(self, user_id, user_input):
        response = self.process_conversation(user_id, user_input)
        self.display_message("Chatbot: " + response)

    def process_image_and_respond(self, user_id, image_path):
        response = self.process_image_with_clip(user_id, image_path)
        self.display_message("Chatbot: " + response)

    def process_image_with_clip(self, user_id, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_embedding = self.clip_model.encode_image(image_input)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            self.image_memory[user_id] = image_embedding.cpu().numpy()
            logger.info(f"Image '{os.path.basename(image_path)}' processed and embedded for User {user_id}.")
            return f"Image '{os.path.basename(image_path)}' processed and embedded for User {user_id}."
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            return f"Failed to process image: {e}"

    def generate_response_with_lstm(self, user_input):
        input_seq = self.tokenizer.texts_to_sequences([user_input])
        input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=self.max_len, padding='post')
        output_probs = self.lstm_model.predict(input_seq)
        response_idx = np.argmax(output_probs, axis=-1)[0]
        response = self.tokenizer.index_word.get(response_idx, "I'm not sure how to respond to that.")
        logger.info(f"LSTM generated response index: {response_idx} -> '{response}'")
        return response.strip()

    def process_conversation(self, user_id, user_input):
        context = get_conversation_context(user_id, max_context=3)
        context_text = " ".join([f"User: {turn['user_input']} Bot: {turn['chatbot_response']}" for turn in context])
        full_input = f"{context_text} User: {user_input}"
        emotion_label, emotion_vector = detect_emotion(full_input)
        state = np.array(emotion_vector)
        knowledge_context_vector = create_and_reason_over_knowledge_graph()
        image_embedding = self.image_memory.get(user_id, np.zeros(128))
        action = self.dqn_agent.act(state, emotion_vector, knowledge_context_vector, image_embedding)
        lstm_response = self.generate_response_with_lstm(user_input)
        affectionate_response = generate_affectionate_response(action, emotion_label)
        response = f"{affectionate_response} {lstm_response}"
        update_conversation_memory(user_id, user_input, response)
        return response

def get_conversation_context(user_id, max_context=3):
    return conversation_memory.get(user_id, [])[-max_context:]

def detect_emotion(text):
    try:
        results = emotion_detector(text)
        emotion_label = results[0]['label'].lower() if results else 'neutral'
        emotion_vector = emotion_to_vector.get(emotion_label, [0, 1, 0])
        logger.info(f"Detected emotion: {emotion_label} with vector {emotion_vector}")
        return emotion_label, emotion_vector
    except Exception as e:
        logger.error(f"Emotion detection failed: {e}")
        return 'neutral', emotion_to_vector['neutral']

def create_and_reason_over_knowledge_graph():
    # Placeholder for your knowledge graph logic
    knowledge_vector = np.random.rand(128)
    logger.info(f"Generated knowledge context vector: {knowledge_vector}")
    return knowledge_vector

def generate_affectionate_response(action, emotion_label):
    responses = {
        0: "Hello! ",
        1: "I'm sorry you're feeling that way. ",
        2: "Stay strong! ",
    }
    emotion_modifiers = {
        "joy": "It's great to hear you're feeling joyful! ",
        "sadness": "I understand things might be tough right now. ",
        "anger": "I sense some frustration. ",
        "fear": "It seems you're feeling uneasy. ",
        "surprise": "That's unexpected! ",
        "disgust": "I'm sorry you're feeling this way. ",
        "neutral": "",
    }
    action_response = responses.get(action, "How can I assist you today? ")
    emotion_modifier = emotion_modifiers.get(emotion_label, "")
    logger.info(f"Generated affectionate response: '{emotion_modifier}{action_response}'")
    return f"{emotion_modifier}{action_response}"

def update_conversation_memory(user_id, user_input, response):
    if user_id not in conversation_memory:
        conversation_memory[user_id] = []
    conversation_memory[user_id].append({'user_input': user_input, 'chatbot_response': response})
    logger.info(f"Updated conversation memory for User {user_id}: {conversation_memory[user_id]}")

def train_lstm_model(new_dataset_path, weights_path, tokenizer_path='tokenizer.pkl', retrain=False):
    if not os.path.exists(new_dataset_path):
        logger.error(f"Dataset file '{new_dataset_path}' not found.")
        sys.exit(1)
        
    df = pd.read_csv(new_dataset_path)
    inputs = df['input'].astype(str).tolist()
    responses = df['response'].astype(str).tolist()

    # Load or create tokenizer
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        logger.info("Tokenizer loaded from file.")
    else:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token='<OOV>')
        tokenizer.fit_on_texts(inputs + responses)
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Tokenizer trained and saved to file.")

    input_sequences = tokenizer.texts_to_sequences(inputs)
    response_sequences = tokenizer.texts_to_sequences(responses)

    max_len = max(len(x) for x in input_sequences + response_sequences)
    input_padded = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_len, padding='post')
    response_padded = tf.keras.preprocessing.sequence.pad_sequences(response_sequences, maxlen=max_len, padding='post')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=max_len),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(10000, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    # Prepare labels: next word prediction
    response_labels = np.array([seq[0] if len(seq) > 0 else 0 for seq in response_sequences])

    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            logger.info("LSTM model weights loaded.")
            if retrain:
                model.fit(input_padded, response_labels, epochs=5, batch_size=32, validation_split=0.2)
                model.save_weights(weights_path)
                logger.info("LSTM model retrained and weights updated.")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            sys.exit(1)
    else:
        model.fit(input_padded, response_labels, epochs=10, batch_size=32, validation_split=0.2)
        model.save_weights(weights_path)
        logger.info("LSTM model trained and weights saved.")

    return model, tokenizer, max_len

def load_and_run_chatbot_gui():
    app = ChatBotApp()
    app.mainloop()

if __name__ == "__main__":
    load_and_run_chatbot_gui()
