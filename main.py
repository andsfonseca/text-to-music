import torch
from pydub import AudioSegment
import numpy as np

from src.utils.training import ModelTrainer
from src.utils.audio import AudioGenerator
 
if __name__ == "__main__":

    final_model_path = "final_autoencoder_model.pth"
    model_name = "autoencoder_model"

    model_trainer = ModelTrainer(model_name, final_model_path)

    model, device = model_trainer.load_model()

    if model == None:
        model_trainer.train()

    print("Generating music...")
    generator = AudioGenerator(model, device)
    
    audio = torch.randn(1, 1, 2**18, device=device)

    generated_audio = generator.generate_music(audio)

    # Save generated audio to a file
    output_filename = f"models/{model_name}/generated_music.mp3"

    sample_rate = 48000  
    generator.save_audio(generated_audio.squeeze(), sample_rate, output_filename)
    print(f"Generated music saved to {output_filename}")  


    

