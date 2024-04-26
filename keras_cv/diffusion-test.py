from keras_cv.models import StableDiffusion
from PIL import Image

model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)
img = model.text_to_image(
    prompt="A beautiful horse running through a field",
    batch_size=1,  # How many images to generate at once
    num_steps=25,  # Number of iterations (controls image quality)
    seed=123,  # Set this to always get the same image from the same prompt
)
Image.fromarray(img[0]).save("horse.png")
print("saved at horse.png")
