import numpy as np
import pandas as pd


fazer_pouco = False
caminho_root = "Código/Dataset/"


#Lê o identity_CelebA.txt
df = pd.read_csv("Código/Dataset/identity_CelebA.txt", sep=" ", names=["img", "id"])

#Conta quantas imagens cada ID tem
contagem = df["id"].value_counts()

#Seleciona o top 20% (ou 5% se eu só queria fazer o pacote para testar)
if fazer_pouco:
    qtd_ids = int(len(contagem) * 0.05)
else:
    qtd_ids = int(len(contagem) * 0.20)


ids_escolhidos = contagem.head(qtd_ids).index

#Filtra o dataframe
df_filtrado = df[df["id"].isin(ids_escolhidos)]

#Deixei em array do numpy para ficar mais fácil de usar em baixo
imgs_filtradas = df_filtrado["img"].to_numpy()
ids_filtrados = df_filtrado["id"].to_numpy()


print(f"total de classes escolhidas: {len(ids_escolhidos)}")
print(f"total de imagens dessas classes: {len(imgs_filtradas)}")


vetores = np.array(imgs_filtradas)
rótulos = np.array(ids_filtrados)


# mapear ids para 0..C-1
ids_unicos = np.unique(rótulos)
mapeamento = {idv: i for i, idv in enumerate(ids_unicos)}
rótulos = np.array([mapeamento[i] for i in rótulos])
num_classes = len(ids_unicos)



#Salvar arquivo que a main vai usar
if fazer_pouco:
    nome_arquivo = f"Código/05Imagens.npz"
else:
    nome_arquivo = f"Código/20Imagens.npz"


np.savez(
    nome_arquivo,
    vetores=vetores,
    rotulos=rótulos,
    ids_unicos=ids_unicos,
)

print(f"\n Arquivo salvo como: {nome_arquivo}")
