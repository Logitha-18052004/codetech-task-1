import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
  
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    return pipe

st.title("Advanced Text-to-Image Generation")

# Load the model
pipe = load_model()

# Move the model to CPU if CUDA is not available
if not torch.cuda.is_available():
    pipe = pipe.to("cpu")
    st.warning("Running on CPU. This may be slow.")
else:
    pipe = pipe.to("cuda")

# Create a text input for the user
prompt = st.text_input("Enter your image description:")

# Create a button to generate the image
if st.button("Generate Image"):
    if prompt:
        try:
            with st.spinner("Generating image... This may take a while."):
                # Generate the image
                image = pipe(prompt).images[0]
            
            # Display the generated image
            st.image(image, caption="Generated Image", use_column_width=True)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a description first.")

st.markdown("Note: This app uses Stable Diffusion v1.5. Image generation may take some time, especially on CPU.")
