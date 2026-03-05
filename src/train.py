from preprocessing import prepare_data
from model import create_model

from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd


# cargar y preparar datos
X_train, X_test, y_train, y_test = prepare_data('../DATOS/clasificacion_del_cancer.csv')


# crear modelo
model = create_model()


# early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=25,
    verbose=1
)


# entrenamiento
history = model.fit(
    x=X_train,
    y=y_train,
    epochs=600,
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[early_stop]
)


# guardar historial para graficar
model_loss = pd.DataFrame(history.history)
model_loss.plot()


# guardar modelo
model.save('../model/cancer_model.h5')
