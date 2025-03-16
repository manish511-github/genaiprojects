from moviepy import *
from pathlib import Path
import speech_recognition as sr
from pytube import YouTube
import yt_dlp
from pprint import pprint
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
from dotenv import load_dotenv
import whisper
import numpy
import matplotlib.pyplot as plt
from logger import logging
import re
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding  
from llama_index.core import StorageContext
from llama_index.core import Settings
from pydantic import Field


from llama_index.vector_stores.lancedb import LanceDBVectorStore


from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode

import google.generativeai as genai
from llama_index.embeddings.gemini import GeminiEmbedding

import torch 
import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")  # Use "gemini-pro-vision" for multimodal inputs

class CLIPEmbedding(BaseEmbedding):
    clip_model: torch.nn.Module = Field(..., description="CLIP model for embeddings")
    clip_preprocess: callable = Field(..., description="CLIP preprocessing function")
    device: str = Field(default="cuda", description="Device to run the model on (e.g., 'cuda' or 'cpu')")

    def __init__(self, clip_model, clip_preprocess, device="cuda"):
        super().__init__(clip_model=clip_model, clip_preprocess=clip_preprocess, device=device)
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device

    def _split_text(self, text, max_length=77):
        """Split text into smaller chunks that fit within CLIP's token limit."""
        sentences = re.split(r'(?<=[.!?]) +', text)  # Split by sentences
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _get_text_embedding(self, text):
        """Embed text by splitting it into smaller chunks and averaging the embeddings."""
        chunks = self._split_text(text)
        embeddings = []
        for chunk in chunks:
            with torch.no_grad():
                text_tokens = clip.tokenize(chunk).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                embeddings.append(text_features.cpu().numpy())
        # Average the embeddings of all chunks and flatten the result
        avg_embedding = numpy.mean(embeddings, axis=0)
        return avg_embedding.flatten().tolist()  # Flatten and convert to list of floats

    def _get_image_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
        return image_features.cpu().numpy().flatten().tolist()  # Flatten and convert to list of floats

    def _get_query_embedding(self, query: str):
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str):
        return self._get_text_embedding(query)

    def get_text_embedding(self, text: str):
        return self._get_text_embedding(text)

    def get_image_embedding(self, image_path: str):
        return self._get_image_embedding(image_path)
    
load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
gemini_embed_model = GeminiEmbedding(model_name="models/multimodal-embedding-gecko")

# print(os.environ.get("OPENAI_API_KEY"))
print(os.getcwd())
video_url = "https://youtu.be/3dhcmeOTZ_Q"
current_working_directory = os.getcwd()
content_folder = os.path.join(current_working_directory, "content")
os.makedirs(content_folder, exist_ok=True)
output_video_folder = os.path.join(content_folder, "video_data")
os.makedirs(output_video_folder, exist_ok=True)
output_video_path = os.path.join(output_video_folder, "output.mp4")
## from the video we are going to collect image, audio, text
print(output_video_path)
output_folder = os.path.join(content_folder, "mixed_data")
os.makedirs(output_folder, exist_ok=True)
print(output_folder)
output_audio_path = os.path.join(output_folder, "mixed_audio_path")
os.makedirs(output_audio_path, exist_ok=True)

def download_video(video_url, output_path):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=mp4]/best[ext=mp4]',  # Ensures MP4 format
        'outtmpl': output_path,  # Set the output file path
        'merge_output_format': 'mp4',
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        metadata = {
            "Author": info.get("uploader"),
            "Title": info.get("title"),
            "Views": info.get("view_count"),
        }
        print("Metadata:", metadata)
    return metadata

metadata_vid = download_video(video_url, output_video_path) 

def video_to_images(video_path, output_path):
    clip= VideoFileClip(video_path)
    clip.write_images_sequence(
        os.path.join(output_path,"frame%04d.png"),fps= 0.2
    )
# video_to_images(output_video_path,output_folder)

def video_to_audio(video_path, output_audio_file):
    # print(video_path)
    # print(output_audio_path)
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(output_audio_file, codec ='mp3')

output_audio_file = os.path.join(output_audio_path, "output_audio.mp3")  # Ensure correct filename
# video_to_audio(output_video_path,output_audio_file)


def audio_to_text (audio_path):
    model = whisper.load_model ("base")
    result = model.transcribe(audio_path)
    return result["text"]

# text_data = audio_to_text(output_audio_file)

# with open(output_folder + "output_text.txt","w") as file :
#     file.write(text_data)
# logging.info("Text data saved to file") 

# os.remove(output_audio_file)
# logging.info ("Audio file removed")

text_store = LanceDBVectorStore(uri= "lancedb", table_name = "text_collection")
image_store = LanceDBVectorStore(uri="lancedb", table_name="image_collection")

storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)


documents = SimpleDirectoryReader(output_folder).load_data()
clip_embed_model = CLIPEmbedding(clip_model, clip_preprocess, device=device)
Settings.embed_model = clip_embed_model
index= MultiModalVectorStoreIndex.from_documents(documents, storage_context=storage_context)

retriever_engine=index.as_retriever(similarity_top_k=5, image_similarity_top_k=5)

def retrieve(retriever_engine, query_str):
    retrieval_results = retriever_engine.retrieve(query_str)

    retrieved_image = []
    retrieved_text = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        else:
            display_source_node(res_node, source_length=200)
            retrieved_text.append(res_node.text)

    return retrieved_image, retrieved_text

query="can you tell me what is linear regression? explain equation of the multiple linear regression?"

img,text=retrieve(retriever_engine,query)

print(img)
# print(text)

def plot_images(images_path):
  images_shown = 0
  plt.figure(figsize=(16, 9))
  for img_path in images_path:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 5:
                break
  plt.show()  # Ensure images are displayed

plot_images(img)

qa_tmpl_str=(
    "Based on the provided information, including relevant images and retrieved context from the video, \
    accurately and precisely answer the query without any additional prior knowledge.\n"

    "---------------------\n"
    "Context: {context_str}\n"
    "Metadata for video: {metadata_str} \n"

    "---------------------\n"
    "Query: {query_str}\n"
    "Answer: "
)
metadata_str = json.dumps(metadata_vid)

query_str= query
context_str = "".join(text)

formatted_query = qa_tmpl_str.format(
    context_str = context_str,
    metadata_str=metadata_str,
    query_str= query_str
)


def query_gemini_with_text_and_images(query, image_paths):
    """Query the Gemini Flash 2.0 model with a text query and multiple images."""
    # Load the images
    images = [Image.open(image_path) for image_path in image_paths]

    # Combine the query and images into a single input list
    inputs = [query] + images

    # Send the query and images to the Gemini Flash 2.0 model
    response = gemini_model.generate_content(inputs)
    return response.text


response = query_gemini_with_text_and_images(formatted_query, img)
print(response)