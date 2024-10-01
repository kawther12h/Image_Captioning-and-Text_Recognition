# Image Captioiong and Text Recognition

The idea of this project is to build a gradio interface for two tasks. First task, image caption extraction, Second task, text recognition with the text is written by hand or digital. The result of these tasks will appear in english and arabic anguage.

## Model and Pipelines

This section will discuss the Model and pipelines used in project.

#### [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-large#blip-bootstrapping-language-image-pre-training-for-unified-vision-language-understanding-and-generation) (Bootstrapping Language-Image Pre-training) --> for image captioning

It is a powerful and versatile multimodal model developed by Salesforce that combines vision and language understanding tasks like image captioning.

The model  pre_trained on a large dataset of image-caption pairs, allowing it to capture a wide range of visual concepts and nuances. It offering high-quality captions, multimodal understanding, and ease of use.
________________________________

#### [Donut](https://huggingface.co/jinhybr/OCR-Donut-CORD) (Document understanding transformer) --> for text extraction

Donut is specifically designed for document understanding tasks, making it well-suited for extracting text from documents with different layouts and formats. This Donut model training on CORD dataset and it has achieved excellent results on various document understanding tasks, including text extraction.

_________________________________

#### [TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten) (base-sized model, fine-tuned on IAM) --> for handwritten text extraction

The model is designed to handle complex handwritten scripts and noisy images, making it suitable for a wide range of real-world scenarios. The model is pre-trained on the IAM dataset, which is a large-scale dataset of handwritten text images. The limitation of this model that you can use the raw model for optical character recognition (OCR) on single text-line images.

_________________________________

#### [Marefa](https://huggingface.co/marefa-nlp/marefa-mt-en-ar)-Mt-En-Ar --> for translation from En to Ar

The model is specifically designed for English-Arabic translation, making it particularly well-suited for this language pair. The special about this model that is take into considration the using of additional Arabic characters like پ or گ.



## Explination of project work

This section will discuss how huggingface image_to_text models used for image captioning and text extraction.

* Install and import the needed library for loading pretrained model, library for gradio interface and other for dealing and processing the image.
```python
# Install needed library
!pip install gradio
!pip install transformers
!pip install torch

# Import needed library
from PIL import Image
import gradio as gr
import torch
import requests
import re
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, TrOCRProcessor, VisionEncoderDecoderModel

```
## Image captioning Function
 * processor_blip: This is a processor for the Blip image captioning model.

   model_blip: The actual Blip image captioning model.

   translate: This model is used for translation from English to Arabic.
```python
processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")

model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

translate = pipeline("translation",model="marefa-nlp/marefa-mt-en-ar")
```
* The ```caption_and_translate``` function takes three arguments:

  img: The input image (presumably in PIL format).

  min_len: Minimum length for the generated caption.

  max_len: Maximum length for the generated caption.
* Inside the function, The input image is converted to RGB format. Then
the Blip model processes the image and generates an English caption.
The English caption is then translated to Arabic using the translate model.
The Arabic caption is formatted with right-to-left directionality. Finally,
both the English and Arabic captions are returned.
```python
def caption_and_translate(img, min_len, max_len):
    raw_image = Image.open(img).convert('RGB')
    inputs_blip = processor_blip(raw_image, return_tensors="pt")
    out_blip = model_blip.generate(**inputs_blip, min_length=min_len, max_length=max_len)
    english_caption = processor_blip.decode(out_blip[0], skip_special_tokens=True)

    arabic_caption = translate(english_caption)
    arabic_caption = arabic_caption[0]['translation_text']

    translated_caption = f'<div dir="rtl">{arabic_caption}</div>'

    return english_caption, translated_caption
```

* The defintion of Gradio interface ```img_cap_en_ar``` for user interaction.
Users can upload an image and adjust the minimum and maximum caption lengths.
The interface displays both English and Arabic captions

### Expected output
![Screenshot 2024-10-01 193547](https://github.com/user-attachments/assets/94b02075-ba34-4073-b393-fbd7148fa399)

## Text Extraction Function
* The code begins by loading two pre-trained models using the pipeline function from the Hugging Face Transformers library.

  The first model is for image-to-text conversion 

  The second model is for translation from English to Arabic
```python
text_rec = pipeline("image-to-text", model="jinhybr/OCR-Donut-CORD")

translate = pipeline("translation",model="marefa-nlp/marefa-mt-en-ar")
```
* The function ```extract_text``` takes an image as input and 
Passes the image to the image-to-text model (text_rec) to extract text.
Removes any HTML tags from the extracted text.
Translates the extracted text from English to Arabic 
Formats the translated text in right-to-left (RTL) direction. Finally,
returns both the original extracted text and the translated text.
```python
def extract_text(image):
    result = text_rec(image)

    text = result[0]['generated_text']
    text = re.sub(r'<[^>]*>', '', text)  # Remove all HTML tags

    arabic_text3 = translate(text)
    arabic_text3 = arabic_text3[0]['translation_text']
    htranslated_text = f'<div dir="rtl">{arabic_text3}</div>'

    return text,htranslated_text
```
* The  Gradio interface ```text_recognition``` for user interaction.
 user can upload an image to the Gradio interface, and it will extract text from the image and display both the original and translated versions.
### Expected output
![Screenshot 2024-10-01 193607](https://github.com/user-attachments/assets/3f485d24-1dd0-4800-97b3-021160bbf42a)

## Handwritten Text Extraction Function
* The code begins by loading two pre-trained models:

  First checkpoint model, the TrOCRProcessor is specifically designed for extracting handwritten text.

  VisionEncoderDecoderModel from the same checkpoint, which is used for encoding and decoding visual information.

  The second model is for translation from English to Arabic.
```python
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')

model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

translate = pipeline("translation",model="marefa-nlp/marefa-mt-en-ar")
```
* The function ```recognize_handwritten_text``` takes an image as input. The image  processed using the (TrOCRProcessor), which converts it into pixel values.The (VisionEncoderDecoderModel) generates IDs for the extracted text. The (processor.batch_decode) method decodes these IDs into actual text then, the   English text is extracted and translated to Arabic. Finally, The function returns both the original extracted English text and Arabic text.
```python
def recognize_handwritten_text(image2):
  pixel_values = processor(images=image2, return_tensors="pt").pixel_values
  generated_ids = model.generate(pixel_values)
  generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

  arabic_text2 = translate(generated_text)
  arabic_text2 = arabic_text2[0]['translation_text']
  htranslated_text = f'<div dir="rtl">{arabic_text2}</div>'

  return generated_text, htranslated_text
```
* The  Gradio interface ```handwritten_rec``` for user interaction.
 user can upload an image to the Gradio interface, and it will extract handwritten text from the image and display both the original and translated versions.
### Expected output
![Screenshot 2024-10-01 193628](https://github.com/user-attachments/assets/4a91a3e1-5a32-45bc-a4d9-d1ddccfb3062)

## Integrate interfaces
* Combine all interfaces into a tabbed interface in Gradio.
```python
demo = gr.TabbedInterface([img_cap_en_ar, text_recognition, handwritten_rec], ["Extract_Caption", " Extract_Digital_text", " Extract_HandWritten_text"])
demo.launch(debug=True)
```
## Project Limitation
* #### In Image Captioning Task: 

  The slider for adjust the minimum and maximum caption lengths works well in Notebook, but for some reasons it doesn't work when uploaded to Huggingface. Therefore, I initialized a static values for minimum and maximum caption lengths 

* #### In Handwritten Text Extraction Task: 

  I expected that the model can handle a multi line document, but it can only handle single line of text.
## Python Notebook of Project
[Image Captioiong and Text Recognition Notebook](https://github.com/kawther12h/Image_Captioning-and-Text_Recognition/blob/main/FinalPro_Image_Captioning_and_Text_Recognition.ipynb)
## Hugging Face project page
[Hugging Face project space](https://huggingface.co/spaces/Kawthar12h/Image_Captioning_Text_Recognition)
## Presentation Slids
[Presentation](https://www.canva.com/design/DAGSWiBC-iw/FOD-uV-PYUe57jGPH_RwTA/view?utm_content=DAGSWiBC-iw&utm_campaign=designshare&utm_medium=link&utm_source=editor)
## Explaination Video
[Explaination Video](https://drive.google.com/file/d/1cSZaDledF1A3vQSbk0S1Y83VorwYcZ1h/view?usp=sharing)


