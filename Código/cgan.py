import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Reshape, Flatten, Embedding, Concatenate,
    Conv2D, Conv2DTranspose, BatchNormalization,
    LeakyReLU, ReLU
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
from PIL import Image


latent_dim = 100
epochs = 50
batch_size = 64
img_shape = (64, 64, 3)

caminho_npz = "Código/20Imagens.npz"
saida_npz   = "Código/20Imagens_cgan.npz"

caminho_imgs = "Código/Dataset/img_align_celeba/"

#carregar
dados = np.load(caminho_npz, allow_pickle=True)
nomes = dados["vetores"]
rotulos = dados["rotulos"]
num_classes = len(dados["ids_unicos"])


def carregar_imagem(nome):
    img = tf.io.read_file(tf.strings.join([caminho_imgs, nome]))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_shape[:2])
    img = (img / 127.5) - 1.0  #[-1,1] para tanh
    return img


imgs = np.array([carregar_imagem(n).numpy() for n in nomes])

dataset = tf.data.Dataset.from_tensor_slices((imgs, rotulos))
dataset = dataset.shuffle(1024).batch(batch_size)








#gerador
def build_generator():
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype="int32")

    label_emb = Embedding(num_classes, latent_dim)(label)
    label_emb = Flatten()(label_emb)

    x = Concatenate()([noise, label_emb])
    x = Dense(8 * 8 * 256, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Reshape((8, 8, 256))(x)

    x = Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    img = Conv2DTranspose(3, 4, strides=2, padding="same", activation="tanh")(x)

    return Model([noise, label], img)



#descriminador
def build_discriminator():
    img = Input(shape=img_shape)
    label = Input(shape=(1,), dtype="int32")

    label_emb = Embedding(num_classes, img_shape[0] * img_shape[1])(label)
    label_emb = Reshape((img_shape[0], img_shape[1], 1))(label_emb)

    x = Concatenate(axis=-1)([img, label_emb])

    x = Conv2D(64, 4, strides=2, padding="same")(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, 4, strides=2, padding="same")(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, 4, strides=2, padding="same")(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    out = Dense(1, activation="sigmoid")(x)

    return Model([img, label], out)



#compila
generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(
    loss="binary_crossentropy",
    optimizer=Adam(0.0002, 0.5),
    metrics=["accuracy"]
)

discriminator.trainable = False #Para o modelo macro não afetar ele



z = Input(shape=(latent_dim,)) #Ruido
lbl = Input(shape=(1,)) #Rotulo
img_fake = generator([z, lbl]) #Geradir
validity = discriminator([img_fake, lbl]) #Discriminador

#modelo macro
cgan = Model([z, lbl], validity)
cgan.compile(
    loss="binary_crossentropy",
    optimizer=Adam(0.0002, 0.5)
)


#treinamento 
for epoch in range(epochs):
    for real_imgs, labels in dataset:
        batch = real_imgs.shape[0]

        valid = np.ones((batch, 1))
        fake = np.zeros((batch, 1))

        ruido = np.random.normal(0, 1, (batch, latent_dim))
        fake_imgs = generator.predict([ruido, labels], verbose=0)

        discriminator.train_on_batch([real_imgs, labels], valid)
        discriminator.train_on_batch([fake_imgs, labels], fake)

        ruido = np.random.normal(0, 1, (batch, latent_dim)) #Gera ruido
        sampled_labels = np.random.randint(0, num_classes, batch) #PEga batch de rotulos

        cgan.train_on_batch([ruido, sampled_labels], valid) #Muda só o gerador porque o peso do discrimandor está travado

    print(f"Epoch {epoch+1}/{epochs} finalizada")



#aqui já começa a realizar a geração das imagens
n_sinteticas = len(imgs) // 2 #só pra gerar 50%

ruido = np.random.normal(0, 1, (n_sinteticas, latent_dim))
labels_sint = np.random.randint(0, num_classes, n_sinteticas)

imgs_fake = generator.predict([ruido, labels_sint], verbose=0)
imgs_fake = (imgs_fake + 1.0) / 2.0  #desnormalizar


pasta_fake = caminho_imgs + "fake_cgan/"
os.makedirs(pasta_fake, exist_ok=True)

#salva
for i, img in enumerate(imgs_fake):
    img_uint8 = (img * 255).astype(np.uint8) #desnormalzar dnv
    Image.fromarray(img_uint8).save(pasta_fake + f"fake_{i}.jpg")

nomes_fake = np.array([f"fake_cgan/fake_{i}.jpg" for i in range(n_sinteticas)])


#salvar npz
vetores_aug = np.concatenate([nomes, nomes_fake])
rotulos_aug = np.concatenate([rotulos, labels_sint])

np.savez(
    saida_npz,
    vetores=vetores_aug,
    rotulos=rotulos_aug,
    ids_unicos=dados["ids_unicos"]
)

print(f"\nDataset aumentado salvo em: {saida_npz}")


#salva os parametros de todos os modelos
generator.save("generator_cgan.keras")
discriminator.save("discriminator_cgan.keras")
cgan.save("cgan_completa.keras")