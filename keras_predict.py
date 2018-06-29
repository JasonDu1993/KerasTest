from keras.models import Model,load_model

model = load_model('model_save_before_fit.h5')
model1 = load_model('model_save_model_function.h5')
# weight = model.load_weights()
print(model.summary())
print('.............')
print(model1.summary())