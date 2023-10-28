from django.shortcuts import render
from django.http import JsonResponse
import tensorflow as tf
import logging
from application import startUp
# Suppress TensorFlow warnings
tf.get_logger().setLevel(logging.ERROR)
transformer, tokenizer_en, tokenizer_ar, MAX_LENGTH, VOCAB_SIZE_EN, VOCAB_SIZE_AR = startUp()


def evaluate(inp_sentence):
    # Tokenize the input sentence
    inp_sequence = tokenizer_en.texts_to_sequences([inp_sentence])[0]
    inp_sentence = [VOCAB_SIZE_EN - 2] + inp_sequence + [VOCAB_SIZE_EN - 1]
    enc_input = tf.expand_dims(inp_sentence, axis=0)

    output = tf.expand_dims([VOCAB_SIZE_AR - 2], axis=0)

    for _ in range(MAX_LENGTH):
        predictions = transformer(enc_input, output, False)  # (1, seq_length, vocab_size_ar)

        prediction = predictions[:, -1:, :]

        predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)

        if predicted_id == VOCAB_SIZE_AR - 1:
            return tf.squeeze(output, axis=0)

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def index(request):
    return render(request, 'EN_AR_APP/index.html')


def translate(sentence):
    output = evaluate(sentence).numpy()
    # Decode the sequence back to text
    predicted_sentence = tokenizer_ar.sequences_to_texts([output])[0]
    # Split the predicted sentence into words and skip the first word, cause it is attached for everything
    words = predicted_sentence.split()[1:]
    # Join the words back into a sentence
    modified_predicted_sentence = ' '.join(words)
    return modified_predicted_sentence


def translate_text(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text')
        translated_text = translate(input_text)  # Assuming this returns the translated text
        return JsonResponse({'translated_text': translated_text})

    return JsonResponse({'error': 'Invalid request method.'})
