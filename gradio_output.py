from transformers import T5ForConditionalGeneration, AutoTokenizer
import gradio as gr
# needs python 3.7 to work !
tokenizer = AutoTokenizer.from_pretrained("vishnun/t5spellcorrector")
model = T5ForConditionalGeneration.from_pretrained('vishnun/t5spellcorrector')


# input_ids = tokenizer.encode('summarize: the blockade was made uz of squadrons of shmps sit uz at different points along the southern coast lyne', return_tensors='pt')
# print(input_ids)

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
                        title='Hi there, I\'m spell checker')

interface.launch(share=True)