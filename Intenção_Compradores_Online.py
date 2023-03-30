import streamlit                as st
import pandas                   as pd
import matplotlib.pyplot        as plt
import seaborn                  as sns
import numpy                    as np

from gower                      import gower_matrix
from scipy.spatial.distance     import pdist, squareform
from scipy.cluster.hierarchy    import linkage, fcluster, dendrogram


df = pd.read_csv('online_shoppers_intention.csv')

@st.cache_data
def elementos_grupo(n):
    df_spotlight[f'grupo_n{n}'] = fcluster(Z, n, criterion = 'maxclust')
    return df_spotlight[[f'grupo_n{n}']].value_counts()
    
 
st.set_page_config(page_title = 'Intenção de Compras dos Compradores Online',
    page_icon = 'images.png', 
    layout = "wide"
)
st.title('Intenção dos Compradores Online')
st.subheader('(Pelo Método Agrupamento Hierárquico)')
st.markdown("---")
st.markdown('''Neste exercício vamos usar a base
    [online shoppers purchase intention](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) 
    de Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018).
    [Web Link](https://doi.org/10.1007/s00521-018-3523-0).
    A base trata de registros de 12.330 sessões de acesso a páginas, cada sessão sendo de um único usuário em um período de
    12 meses, para posteriormente estudarmos a relação entre o design da página e o perfil do cliente - 
    "Será que clientes com comportamento de navegação diferentes possuem propensão a compra diferente?" 
    Nosso objetivo agora é agrupar as sessões de acesso ao portal considerando o comportamento de acesso e
    informações da data, como a proximidade a uma data especial, fim de semana e o mês.'''
)
st.markdown('  ')
st.markdown('  ')
st.markdown('''
    |Variavel                |Descrição          | 
    |------------------------|:-------------------| 
    |Administrative          | Quantidade de acessos em páginas administrativas| 
    |Administrative_Duration | Tempo de acesso em páginas administrativas | 
    |Informational           | Quantidade de acessos em páginas informativas  | 
    |Informational_Duration  | Tempo de acesso em páginas informativas  | 
    |ProductRelated          | Quantidade de acessos em páginas de produtos | 
    |ProductRelated_Duration | Tempo de acesso em páginas de produtos | 
    |BounceRates             | *Percentual de visitantes que entram no site e saem sem acionar outros *requests* durante a sessão  | 
    |ExitRates               | * Soma de vezes que a página é visualizada por último em uma sessão dividido pelo total de visualizações | 
    |PageValues              | * Representa o valor médio de uma página da Web que um usuário visitou antes de concluir uma transação de comércio eletrônico | 
    |SpecialDay              | Indica a proximidade a uma data festiva (dia das mães etc) | 
    |Month                   | Mês  | 
    |OperatingSystems        | Sistema operacional do visitante | 
    |Browser                 | Browser do visitante | 
    |Region                  | Região | 
    |TrafficType             | Tipo de tráfego                  | 
    |VisitorType             | Tipo de visitante: novo ou recorrente | 
    |Weekend                 | Indica final de semana | 
    |Revenue                 | Indica se houve compra ou não |
    \* variávels calculadas pelo google analytics'''
)
st.markdown('  ')
st.markdown('  ')
st.markdown( '### DataFrame: online_shoppers_intention.csv (10 primeiras linhas)')
st.dataframe(df.head(10))
st.markdown('  ')
st.markdown('  ')
st.markdown('Total de números de linhas: 12.330')
st.markdown('Total de números de colunas: 18')
st.markdown('  ')
st.markdown('  ')
st.markdown( '## Variáveis de Agrupamento')
st.markdown('  ')
st.markdown('  ')

df1 = df.loc[:,'Administrative':'ProductRelated_Duration'] # selecionando o pradão de navegação na sessão.
df2 = df.loc[:,'SpecialDay': 'Month'] #selecionando variáveis que indicam a característica da data.
df2 = pd.get_dummies(df2, drop_first=False) # tratamento especial para variáveis qualitaivas.
df_spotlight = pd.concat([df1, df2], axis=1)

st.dataframe(df_spotlight.head())
st.write('''Variáveis que descrevem o padrão da navegação da sessão são: Administrative, Administrative_Duration, 
    Informational, Informational_Duration, ProductRelated, ProductRlated_Duration'''
)
st.markdown('Total de números de linhas: 12.330')
st.markdown('Total de números de colunas: 17')
st.markdown('  ')
st.markdown('  ')
st.markdown( '## Números de Grupos e Avaliação dos Grupos')
st.markdown('  ')
st.markdown('  ')

st.markdown('''Utilizando o Método de Agrupamento Hierárquico (Perform hierarchical/agglomerative clustering),
            avaliamos a distribuição de elementos para agrupamentos em 3 grupos e para 4 grupos.''')
st.markdown('  ')

vars_cat = [True if x in {'Month_Aug','Month_Dec','Month_Feb', 'Month_Jul', 'Month_June', 'Month_Mar', 'Month_May', 'Month_Nov',
                          'Month_Oct', 'Month_Sep'} else False for x in df_spotlight.columns]
distancia_gower = gower_matrix(df_spotlight, cat_features = vars_cat)
gdv = squareform(distancia_gower, force = 'tovector')
Z = linkage(gdv, method = 'complete')
Z_df = pd.DataFrame(Z, columns = ['id1', 'id2', 'dist', 'n'])
st.dataframe(Z_df.head())
st.markdown('  ')
st.markdown('Para o conjunto de 3 grupos, obte-se a seguinte quantidade de elementos:')
st.markdown('  ')
st.dataframe(pd.DataFrame(elementos_grupo(3)))
st.markdown('  ')
st.markdown('Para o conjunto de 4 grupos, obte-se a seguinte quantidade de elementos:')
st.dataframe(pd.DataFrame(elementos_grupo(4)))
st.markdown('  ')
st.markdown( '### Tabelas Cruzadas - Comparativos para 3 Grupos e 4 Grupos')
st.markdown('  ')

df_spotlight.loc[df_spotlight['Administrative']>0, 'Types_of_access']='Administrative'
df_spotlight.loc[df_spotlight['Informational']>0, 'Types_of_access']='Informational'
df_spotlight.loc[df_spotlight['ProductRelated']>0, 'Types_of_access']='ProductRelated'

df_spotlight.loc[df_spotlight['Month_Aug']==1, 'Month']='August'
df_spotlight.loc[df_spotlight['Month_Dec']==1, 'Month']='December'
df_spotlight.loc[df_spotlight['Month_Feb']==1, 'Month']='February'
df_spotlight.loc[df_spotlight['Month_Jul']==1, 'Month']='July'
df_spotlight.loc[df_spotlight['Month_June']==1, 'Month']='June'
df_spotlight.loc[df_spotlight['Month_Mar']==1, 'Month']='March'
df_spotlight.loc[df_spotlight['Month_May']==1, 'Month']='May'
df_spotlight.loc[df_spotlight['Month_Nov']==1, 'Month']='November'
df_spotlight.loc[df_spotlight['Month_Oct']==1, 'Month']='October'
df_spotlight.loc[df_spotlight['Month_Sep']==1, 'Month']='September'

st.dataframe(pd.crosstab(df_spotlight['Types_of_access'], df_spotlight['grupo_n3']))
st.markdown('  ')
st.dataframe(pd.crosstab(df_spotlight['Types_of_access'], df_spotlight['grupo_n4']))
st.markdown('  ')
st.dataframe(pd.crosstab(df_spotlight['Month'], df_spotlight['grupo_n3']))
st.markdown('  ')
st.dataframe(pd.crosstab(df_spotlight['Month'], df_spotlight['grupo_n4']))
st.markdown('  ')
st.dataframe(pd.crosstab([df_spotlight['Month'],df_spotlight['SpecialDay']], 
                         df_spotlight['grupo_n4']))
st.markdown('  ')
st.write('''Analisando as tabelas cruzadas, verifica-se que o melhor agrupamento é o de 4 grupos, visto que o grupo
 de 4 representa o desmenbramento do grupo 3 do agrupamento grupo_n3. Enfim, o agrupamento de 4 grupos está melhor 
 distribuído.''')
st.write('''Por outro lado, não é possível sugerir nomes aos grupos, visto que o grupo 1 representa acessos de MAio
e Dezembro, que em grande parte também ocorreu no grupo 3 e 4.''')
st.write(''' Observa-se contudo que o grupo 2 representa com toda certeza acessos ocorridos em Agosto e Novembro (seja
Administrativo, Informativo ou Relativo ao Produto), uma vez que hoive acessos nestes meses para os grupos 1, 3 e 4''')
st.markdown('  ')
st.markdown('  ')
st.markdown( '### Avaliação de resultados')    
df_spotlight = pd.concat([df_spotlight,df[['BounceRates', 'Revenue']]], axis=1)
df_spotlight.head()
pd.crosstab(df_spotlight['BounceRates'], df_spotlight['grupo_n4'])
compra = pd.crosstab(df_spotlight['Revenue'], df_spotlight['grupo_n4'])
st.dataframe(compra)
st.write('''O grupo que possui clientes mais propensos à compra é o grupo 2, que 
         representa clientes que acessaram no site nos meses de Agosto e Novembro''')