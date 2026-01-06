import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
import datetime
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense
)


#funções auxiliares para inicializar pesos
def inicializar_weights_he(inp,out): #He initialization
    return np.random.randn(out,inp) * np.sqrt(2.0 / inp)

def inicializar_weights_xavier(inp,out): #Xavier/Glorot initialization
    return np.random.randn(out,inp) * np.sqrt(2.0 / (inp + out))


def processar(nome, rotulo):
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

dataset = tf.data.Dataset.from_tensor_slices((nomes_imgs, rotulos))
dataset = dataset.map(processar, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])



























#ESCRITA e Parametros


            precisoes.append(precisao)
            recalls.append(recall)
            f1_scores.append(f1)

            print(f"Acurácia: {acur:.4f}")
            print(f"Precisão: {precisao:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")

            #melhor e pior fold
            if acur > melhor_fold[1]:
                melhor_fold = (fold_id,acur)
            if acur < pior_fold[1]:
                pior_fold = (fold_id,acur)


        #resultados finais do modelo
        print(f"\nAcurácia média: {np.mean(acuracias):.4f} ± {np.std(acuracias):.4f}")
        print(f"Melhor fold: {melhor_fold}")
        print(f"Pior fold: {pior_fold}")


        #salva configuração final
        with open(arquivo_config,"w",encoding="utf-8") as f:
            f.write(f"EXECUTION_TIMESTAMP: {timestamp}\n")
            f.write(f"DESCRIPTOR: {descritor}\n")
            f.write(f"MODEL: {modelo}\n")
            f.write(f"GLOBAL_ACURACY: {np.mean(acuracias):.4f} \n\n")

            if modelo == "linear":
                f.write(f"LINEAR_SPECIFICATION: ('input_layer',{n_atrib},'softmax','cross_entropy')\n")
                f.write(f"LINEAR_OPERATION_LR_METHOD: FIX\n")
                f.write(f"LINEAR_OPERATION_LR_PARAMS: {lr}\n")
                f.write(f"LINEAR_OPERATION_INITIALISATION: Glorot_Bengio_2010\n")
                f.write(f"LINEAR_OPERATION_MAX_EPOCHS: {epocas}\n")
                f.write(f"LINEAR_OPERATION_BATCH_SIZE: {batch}\n")
                f.write(f"LINEAR_OPERATION_PATIENCE: {paciencia}\n")
                f.write(f"LINEAR_OPERATION_L2: {l2}\n")
            else:
                f.write(f"MLP_SPECIFICATION: ('layer 0',{h1},'relu','cross_entropy')\n")
                f.write(f"MLP_SPECIFICATION: ('layer 1',{h2},'relu','cross_entropy')\n")
                f.write(f"MLP_SPECIFICATION: ('output_layer',{num_classes},'softmax','cross_entropy')\n")
                f.write(f"MLP_OPERATION_LR_METHOD: FIX\n")
                f.write(f"MLP_OPERATION_LR_PARAMS: {lr}\n")
                f.write(f"MLP_OPERATION_INITIALISATION: He_2015\n")
                f.write(f"MLP_OPERATION_MAX_EPOCHS: {epocas}\n")
                f.write(f"MLP_OPERATION_MIN_EPOCHS: 1\n")
                f.write(f"MLP_OPERATION_STOP_WINDOW: {paciencia}\n")
                f.write(f"MLP_OPERATION_BATCH_SIZE: {batch}\n")
                f.write(f"MLP_OPERATION_L2: {l2}\n")
                f.write(f"MLP_OPERATION_DROPOUT_RATE: {dropout_rate}\n")


        with open(arquivo_dat,"w",encoding="utf-8") as f:
            if modelo == "linear":
            
                f.write("MODEL: LINEAR\n")
                f.write(f"INPUT_DIM: {n_atrib}\n")
                f.write(f"NUM_CLASSES: {num_classes}\n")
                f.write(f"LR: {lr}\n")
                f.write(f"L2: {l2}\n\n")

                f.write("WEIGHTS\n")
                for row in W:
                    f.write(" ".join(f"{v:.8f}" for v in row) + "\n")

                f.write("\nBIAS\n")
                f.write(" ".join(f"{v:.8f}" for v in b))

            else:
                f.write("MODEL: MLP\n")
                f.write(f"INPUT_DIM: {n_atrib}\n")
                f.write(f"HIDDEN_LAYER_1: {h1}\n")
                f.write(f"HIDDEN_LAYER_2: {h2}\n")
                f.write(f"NUM_CLASSES: {num_classes}\n")
                f.write(f"LR: {lr}\n")
                f.write(f"L2: {l2}\n")
                f.write(f"DROPOUT_RATE: {dropout_rate}\n\n")

                f.write("WEIGHTS_LAYER_1\n")
                for row in W1:
                    f.write(" ".join(f"{v:.8f}" for v in row) + "\n")

                f.write("\nBIAS_LAYER_1\n")
                f.write(" ".join(f"{v:.8f}" for v in b1) + "\n")

                f.write("\nWEIGHTS_LAYER_2\n")
                for row in W2:
                    f.write(" ".join(f"{v:.8f}" for v in row) + "\n")

                f.write("\nBIAS_LAYER_2\n")
                f.write(" ".join(f"{v:.8f}" for v in b2) + "\n")

                f.write("\nWEIGHTS_OUTPUT_LAYER\n")
                for row in W3:
                    f.write(" ".join(f"{v:.8f}" for v in row) + "\n")

                f.write("\nBIAS_OUTPUT_LAYER\n")
                f.write(" ".join(f"{v:.8f}" for v in b3))


