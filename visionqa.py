from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
from keras.layers import add, concatenate

# First, let's define a vision model using a Sequential model.
# This model will encode an image into a vector.
vision_model = Sequential()
vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Flatten())
print('summary', vision_model.summary())

# Now let's get a tensor with the output of our vision model:
image_input = Input(shape=(224, 224, 3))
encoded_image = vision_model(image_input)
print('vision_model output', vision_model.get_output_at(0))
print('encoded_image', encoded_image)

# Next, let's define a language model to encode the question into a vector.
# Each question will be at most 100 word long,
# and we will index words as integers from 1 to 9999.
question_input = Input(shape=(100,), dtype='int32')
print('question_input', question_input)
embedded_question = Embedding(input_dim=10000, output_dim=256)(question_input)
print('embedded_question', embedded_question)
encoded_question = LSTM(256)(embedded_question)
print('encoded_question', encoded_question)

# Let's concatenate the question vector and the image vector:
merged = concatenate([encoded_question, encoded_image])
print('merged', merged)
# And let's train a logistic regression over 1000 words on top:
output = Dense(1000, activation='softmax')(merged)

# This is our final model:
vqa_model = Model(inputs=[image_input, question_input], outputs=output)

# The next stage would be training this model on actual data.

# 视频问答模型
from keras.layers import TimeDistributed

video_input = Input(shape=(101, 256, 256, 3))
print('video_input', video_input)
# This is our video encoded via the previously trained vision_model (weights are reused)
encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # the output will be a sequence of vectors
# t = TimeDistributed(Conv2D(64, (3, 3), padding='same'))(video_input)
# print('t', t)
print('encoded_frame_sequence', encoded_frame_sequence)
encoded_video = LSTM(312)(encoded_frame_sequence)  # the output will be a vector
print('encoded_video', encoded_video)
# This is a model-level representation of the question encoder, reusing the same weights as before:
question_encoder = Model(inputs=question_input, outputs=encoded_question, name='question_encoder')
print('question_encoder', question_encoder.summary())
# Let's use it to encode the question:
video_question_input = Input(shape=(100,), dtype='int32')
print('video_question_input', video_question_input)
encoded_video_question = question_encoder(video_question_input)
print('encoded_video_question', encoded_video_question)
# And this is our video question answering model:
merged = concatenate([encoded_video, encoded_video_question])
print('merged', merged)
output = Dense(1000, activation='softmax')(merged)
print('output', output)
video_qa_model = Model(inputs=[video_input, video_question_input], outputs=output, name='video_qa_model')
print('video_qa_model', video_qa_model.summary())
