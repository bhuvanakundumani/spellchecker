from transformers import T5ForConditionalGeneration, AutoTokenizer
# needs python 3.7 to work ! Uncomment gradio related commands to deploy on gradio
import gradio as gr

model_folder = 'model_dec28'
tokenizer = AutoTokenizer.from_pretrained(model_folder)
model = T5ForConditionalGeneration.from_pretrained(model_folder)

def correct(inputs):
    input_ids = tokenizer.encode(inputs,return_tensors='pt')
    sample_output = model.generate(
        input_ids,
        do_sample=True,
        max_length=50,
        top_p=0.99,
        num_return_sequences=1
    )
    res = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    return res


app_inputs = gr.inputs.Textbox(lines=3, placeholder="Enter a grammatically incorrect sentence here...")

interface = gr.Interface(fn=correct,
                        inputs=app_inputs,
                        outputs='text',
                        title='Welcome to spell checker using T5')

interface.launch(share=True)