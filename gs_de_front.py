import pandas as pd
from prophet import Prophet
import streamlit as st

# Carregar os dados
@st.cache
def load_data():
    return pd.read_csv('dados_tratados_consumo.csv')

data = load_data()

# Configurar a aplicação Streamlit
st.title("Previsão de Consumo por Estado")
st.write("Selecione um estado para visualizar a previsão de consumo para os três meses seguintes ao final dos dados disponíveis.")

# Selecionar o estado
states = data['UF'].unique()
selected_state = st.selectbox("Selecione o estado:", states)

# Filtrar os dados pelo estado selecionado
state_data = data[data['UF'] == selected_state]

# Preparar os dados para o Prophet
state_data['ds'] = pd.to_datetime(state_data['Ano'].astype(int).astype(str) + '-' + state_data['Mes'], format='%Y-%b', errors='coerce')
state_data = state_data.dropna(subset=['ds'])
state_data = state_data.rename(columns={'Consumo': 'y'})[['ds', 'y']]

# Verificar se há dados suficientes
if state_data.shape[0] < 2:
    st.write("Dados insuficientes para realizar a previsão para este estado.")
else:
    # Treinar o modelo Prophet
    model = Prophet()
    model.fit(state_data)

    # Criar um futuro dataframe para três meses após o final dos dados
    future = model.make_future_dataframe(periods=3, freq='M')
    forecast = model.predict(future)

    # Filtrar os últimos três meses previstos
    last_date = state_data['ds'].max()
    forecast_tail = forecast[forecast['ds'] > last_date]

    # Mostrar os resultados
    st.write(f"Previsão para o estado: {selected_state}")
    st.line_chart(forecast[['ds', 'yhat']].set_index('ds'))

    # Mostrar os dados previstos com maior legibilidade
    st.write("Previsão para os três meses seguintes:")
    st.dataframe(forecast_tail[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
