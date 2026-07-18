import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
# Importa o tradutor do deep-translator
from deep_translator import GoogleTranslator

# Baixa o léxico do VADER
nltk.download('vader_lexicon')

st.title("Analisador de Sentimento (PT-BR)")

# Entrada de texto em português
texto_usuario = st.text_area("Digite seu texto em português:")

if st.button("Analisar"):
    if texto_usuario:
        # Traduz o texto de português (pt) para inglês (en)
        texto_ingles = GoogleTranslator(source='pt', target='en').translate(texto_usuario)
        
        # Inicializa o analisador do VADER
        sia = SentimentIntensityAnalyzer()
        
        # Analisa o texto já traduzido
        scores = sia.polarity_scores(texto_ingles)
        compound = scores['compound']
        
        # Classifica o sentimento com base no score
        if compound >= 0.05:
            sentimento = "Positivo"
        elif compound <= -0.05:
            sentimento = "Negativo"
        else:
            sentimento = "Neutro"
        
        # Exibe os resultados na tela
        st.write(f"**Texto traduzido para análise:** *{texto_ingles}*")
        st.write(f"**Sentimento:** {sentimento}")
        st.write(f"**Pontuação (Compound):** {compound}")
    else:
        st.write("Por favor, insira um texto para analisar.")