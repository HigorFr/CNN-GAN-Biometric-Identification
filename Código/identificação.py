import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D,MaxPooling2D,Flatten,Dense)
from sklearn.metrics import ( precision_score, recall_score, f1_score, accuracy_score)

#funções auxiliares para inicializar pesos
def inicializar_weights_he(inp,out): #He initialization
    return np.random.randn(out,inp) * np.sqrt(2.0 / inp)

def inicializar_weights_xavier(inp,out): #Xavier/Glorot initialization
    return np.random.randn(out,inp) * np.sqrt(2.0 / (inp + out))


def preprocessar(nome, rotulo):
    img = tf.io.read_file(tf.strings.join([caminho_root, nome]))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (128, 128))
    img = img / 255.0
    return img, tf.one_hot(rotulo, num_classes)


usar_pouco = False #Só para mudar se eu quero testar com menos dados

if usar_pouco:
    dados = np.load("Código/20Imagens.npz", allow_pickle=True)
else:
    dados = np.load("Código/05Imagens.npz", allow_pickle=True)



nomes_imgs = dados["vetores"]
rotulos = dados["rotulos"]
num_classes = len(dados["ids_unicos"])


#configurações gerais
timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
random_state = 42               #seed para reproducibilidade


caminho_root = "Código/Dataset/img_align_celeba/"



# total de amostras
n_total = len(nomes_imgs)
n_treino = int(0.70 * n_total)
n_val    = int(0.15 * n_total)


dataset = tf.data.Dataset.from_tensor_slices((nomes_imgs, rotulos))
dataset = dataset.shuffle(n_total, seed=random_state) #Embaralha só para garantir


#quebra em 70 15 15 conforme pedido
ds_treino = dataset.take(n_treino)
ds_resto   = dataset.skip(n_treino)
ds_val  = ds_resto.take(n_val)
ds_test = ds_resto.skip(n_val)

#Aplica a função e já organiza os dataset
ds_treino = ds_treino.map(preprocessar).batch(32).prefetch(tf.data.AUTOTUNE)
ds_val    = ds_val.map(preprocessar).batch(32).prefetch(tf.data.AUTOTUNE)
ds_test   = ds_test.map(preprocessar).batch(32).prefetch(tf.data.AUTOTUNE)




model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


#Aplica já com validação
history = model.fit(
    ds_treino,
    epochs=20,
    validation_data=ds_val
)


y_true = []
y_pred = []

for x, y in ds_test:
    preds = model.predict(x)
    y_true.extend(np.argmax(y.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

#cm = confusion_matrix(y_true, y_pred)
#
#disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#disp.plot(cmap='Blues')
#plt.title("Matriz de Confusão – CNN Facial")
#plt.show()



acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='macro')
rec  = recall_score(y_true, y_pred, average='macro')
f1   = f1_score(y_true, y_pred, average='macro')

print("\nMétricas no conjunto de teste:")
print(f"Acurácia : {acc:.4f}")
print(f"Precisão : {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")


plt.figure()
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.xlabel('Época')
plt.ylabel('Loss (Cross-Entropy)')
plt.title('Erro de Treino vs Validação')
plt.legend()
plt.grid(True)
plt.show()



with open("metricas_teste.txt", "w", encoding="utf-8") as f:
    f.write("RESULTADOS NO CONJUNTO DE TESTE\n")
    f.write("--------------------------------\n")
    f.write(f"Acurácia : {acc:.6f}\n")
    f.write(f"Precisão : {prec:.6f}\n")
    f.write(f"Recall   : {rec:.6f}\n")
    f.write(f"F1-score : {f1:.6f}\n")


with open("historico_treino.txt", "w", encoding="utf-8") as f:
    f.write("EPOCA\tLOSS_TREINO\tLOSS_VAL\tACUR_TREINO\tACUR_VAL\n")

    for i in range(len(history.history['loss'])):
        f.write(
            f"{i+1}\t"
            f"{history.history['loss'][i]:.6f}\t"
            f"{history.history['val_loss'][i]:.6f}\t"
            f"{history.history['accuracy'][i]:.6f}\t"
            f"{history.history['val_accuracy'][i]:.6f}\n"
        )

