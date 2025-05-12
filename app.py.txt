# app.py
import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np
import plotly.graph_objects as go # Usaremos go diretamente para mais controle
from datetime import timedelta

# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title="Estratégia EMA RSI Diário",
    page_icon="📈",
    layout="wide" # Usa mais espaço da tela
)

# --- Função do Backtest (Adaptada do código anterior) ---
# Usar cache para evitar recalcular tudo a cada interação pequena
@st.cache_data(ttl=3600) # Cache por 1 hora
def run_strategy_backtest(ticker: str, period_days: int = 365, initial_capital: float = 1000.0, rsi_len: int = 2, ema_len: int = 2, exit_days: int = 4):
    """
    Executa o backtest da estratégia EMA RSI Diário.
    Retorna um dicionário com resultados ou None em caso de erro.
    """
    st.write(f"Buscando dados para {ticker}...") # Feedback para o usuário
    # --- 1. Buscar Dados Históricos ---
    start_date = (pd.Timestamp.today() - timedelta(days=period_days + 50)).strftime('%Y-%m-%d')
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
        if data.empty:
            st.error(f"Não foi possível obter dados para {ticker} no período solicitado.")
            return None
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    except Exception as e:
        st.error(f"Erro ao buscar dados para {ticker}: {str(e)}")
        return None

    st.write(f"Calculando indicadores...") # Feedback
    # --- 2. Calcular Indicadores ---
    try:
        data.ta.rsi(close='Close', length=rsi_len, append=True, col_names=(f'RSI_{rsi_len}_C',))
        data.ta.rsi(close='Open', length=rsi_len, append=True, col_names=(f'RSI_{rsi_len}_O',))
        rsi_close_col = f'RSI_{rsi_len}_C'
        rsi_open_col = f'RSI_{rsi_len}_O'
        data.ta.ema(close=rsi_close_col, length=ema_len, append=True, col_names=(f'EMA_{ema_len}_RSI_C',))
        data.ta.ema(close=rsi_open_col, length=ema_len, append=True, col_names=(f'EMA_{ema_len}_RSI_O',))
        ema_rsi_c_col = f'EMA_{ema_len}_RSI_C'
        ema_rsi_o_col = f'EMA_{ema_len}_RSI_O'
        data.dropna(inplace=True)
        if data.empty:
             st.error(f"Não há dados suficientes para {ticker} após cálculo dos indicadores.")
             return None
    except Exception as e:
        st.error(f"Erro ao calcular indicadores para {ticker}: {str(e)}")
        return None

    st.write(f"Executando simulação...") # Feedback
    # --- 3. Simulação / Lógica de Backtesting ---
    cash = initial_capital
    equity = initial_capital
    shares = 0.0
    in_position = False
    entry_price = 0.0
    entry_date = None
    days_in_trade = 0
    trades = []
    signals = []
    equity_curve = []

    for i in range(1, len(data)):
        current_date = data.index[i]
        current_open_price = data['Open'].iloc[i]
        prev_row = data.iloc[i-1]
        prev_ema_rsi_c = prev_row[ema_rsi_c_col]
        prev_ema_rsi_o = prev_row[ema_rsi_o_col]
        prev_prev_ema_rsi_c = data[ema_rsi_c_col].iloc[i-2] if i > 1 else np.nan
        prev_prev_ema_rsi_o = data[ema_rsi_o_col].iloc[i-2] if i > 1 else np.nan

        exit_signal = False
        exit_reason = None
        if in_position:
            days_in_trade += 1
            is_crossunder = prev_ema_rsi_c < prev_ema_rsi_o and prev_prev_ema_rsi_c >= prev_prev_ema_rsi_o
            is_ema_falling = prev_ema_rsi_c < prev_prev_ema_rsi_c
            if is_crossunder and is_ema_falling:
                exit_signal = True
                exit_reason = "Saída Técnica (Crossunder)"
            elif days_in_trade == exit_days:
                exit_signal = True
                exit_reason = f"Saída por Tempo ({exit_days} dias)"

            if exit_signal:
                exit_price = current_open_price
                if shares > 0: # Só processa se tiver ações
                    cash += shares * exit_price
                    profit = (exit_price - entry_price) * shares
                    return_pct = ((exit_price / entry_price) - 1) * 100 if entry_price != 0 else 0
                    trades.append({
                        "entryDate": entry_date.strftime('%Y-%m-%d'), "entryPrice": round(entry_price, 2),
                        "exitDate": current_date.strftime('%Y-%m-%d'), "exitPrice": round(exit_price, 2),
                        "shares": round(shares, 4), "profit": round(profit, 2),
                        "returnPercent": round(return_pct, 2), "exitReason": exit_reason, "daysHeld": days_in_trade
                    })
                    signals.append({"date": current_date.strftime('%Y-%m-%d'), "type": "sell", "price": round(exit_price, 2)})
                shares = 0.0
                in_position = False
                entry_price = 0.0
                entry_date = None
                days_in_trade = 0

        if not in_position and i > 1:
            is_crossover = prev_ema_rsi_c > prev_ema_rsi_o and prev_prev_ema_rsi_c <= prev_prev_ema_rsi_o
            is_ema_rising = prev_ema_rsi_c > prev_prev_ema_rsi_c
            if is_crossover and is_ema_rising:
                entry_price = current_open_price
                if entry_price > 0 and cash > 0:
                    shares_to_buy = cash / entry_price
                    cash -= shares_to_buy * entry_price
                    shares = shares_to_buy
                    in_position = True
                    entry_date = current_date
                    days_in_trade = 0
                    signals.append({"date": current_date.strftime('%Y-%m-%d'), "type": "buy", "price": round(entry_price, 2)})

        current_equity = cash + (shares * data['Close'].iloc[i])
        equity_curve.append({"date": current_date.strftime('%Y-%m-%d'), "equity": round(current_equity, 2)})

    # --- 4. Calcular Resultados Finais ---
    final_equity = equity_curve[-1]['equity'] if equity_curve else initial_capital
    total_profit = final_equity - initial_capital
    total_return_percent = (total_profit / initial_capital) * 100 if initial_capital != 0 else 0
    number_of_trades = len(trades)
    chart_data_df = data[['Open', 'High', 'Low', 'Close']].reset_index() # Manter como DataFrame
    chart_data_df = chart_data_df.tail(period_days) # Limitar aos dias do backtest

    # --- 5. Retornar Resultados ---
    results = {
        "ticker": ticker,
        "metrics": {
            "initialCapital": round(initial_capital, 2), "finalEquity": round(final_equity, 2),
            "totalProfit": round(total_profit, 2), "totalReturnPercent": round(total_return_percent, 2),
            "numberOfTrades": number_of_trades,
            "startDate": chart_data_df['Date'].min().strftime('%Y-%m-%d') if not chart_data_df.empty else 'N/A',
            "endDate": chart_data_df['Date'].max().strftime('%Y-%m-%d') if not chart_data_df.empty else 'N/A',
            "backtestPeriodDays": period_days,
        },
        "trades": pd.DataFrame(trades) if trades else pd.DataFrame(), # Converter para DataFrame
        "signals": signals,
        "chartData": chart_data_df
    }
    return results

# --- Função para Plotar o Gráfico ---
def plot_results(chart_data_df, signals, ticker):
    if chart_data_df.empty:
        return go.Figure()

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(x=chart_data_df['Date'],
                               open=chart_data_df['Open'],
                               high=chart_data_df['High'],
                               low=chart_data_df['Low'],
                               close=chart_data_df['Close'],
                               name='OHLC'))

    # Sinais de Compra
    buy_signals_df = pd.DataFrame([s for s in signals if s['type'] == 'buy'])
    if not buy_signals_df.empty:
        fig.add_trace(go.Scatter(x=buy_signals_df['date'],
                                 y=buy_signals_df['price'],
                                 mode='markers', name='Compra',
                                 marker=dict(color='green', size=10, symbol='triangle-up')))

    # Sinais de Venda
    sell_signals_df = pd.DataFrame([s for s in signals if s['type'] == 'sell'])
    if not sell_signals_df.empty:
        fig.add_trace(go.Scatter(x=sell_signals_df['date'],
                                 y=sell_signals_df['price'],
                                 mode='markers', name='Venda',
                                 marker=dict(color='red', size=10, symbol='triangle-down')))

    fig.update_layout(
        title=f'Backtest: {ticker}',
        xaxis_title='Data',
        yaxis_title='Preço',
        xaxis_rangeslider_visible=False, # Esconde o range slider
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- Interface do Streamlit ---
st.title("📈 Simulador de Estratégia EMA RSI Diário")
st.markdown("Teste a estratégia de cruzamento de EMA(2) sobre RSI(2) diário.")

# Input do Ticker
ticker_input = st.text_input("Digite o Código do Ativo (ex: PETR4.SA, AAPL, BTC-USD):", value="PETR4.SA").upper()

# Botão para simular
if st.button("Simular Estratégia"):
    if ticker_input:
        with st.spinner(f"Executando backtest para {ticker_input}... Aguarde!"):
            # Chama a função de backtest
            results = run_strategy_backtest(ticker=ticker_input)

            # Armazena os resultados na sessão para evitar perdê-los se houver interações
            st.session_state['backtest_results'] = results
            st.session_state['last_ticker'] = ticker_input # Guarda qual ticker foi simulado
    else:
        st.warning("Por favor, insira um código de ativo.")
        st.session_state['backtest_results'] = None # Limpa resultados antigos

# --- Exibição dos Resultados ---
# Verifica se há resultados na sessão E se pertencem ao último ticker simulado
if 'backtest_results' in st.session_state and st.session_state['backtest_results'] is not None:
    results = st.session_state['backtest_results']

    st.success(f"Backtest concluído para {results['ticker']}")

    # Seção de Métricas
    st.subheader("📊 Métricas Principais")
    metrics = results['metrics']
    col1, col2, col3 = st.columns(3)
    col1.metric("Rentabilidade Total", f"{metrics['totalReturnPercent']:.2f}%")
    col2.metric("Lucro/Prejuízo Total", f"{metrics['totalProfit']:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")) # Format PT-BR
    col3.metric("Nº de Trades", metrics['numberOfTrades'])

    col4, col5, col6 = st.columns(3)
    col4.metric("Capital Inicial", f"{metrics['initialCapital']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    col5.metric("Capital Final", f"{metrics['finalEquity']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    col6.metric("Período", f"{metrics['startDate']} a {metrics['endDate']}")

    # Seção do Gráfico
    st.subheader("📈 Gráfico de Velas com Sinais")
    fig = plot_results(results['chartData'], results['signals'], results['ticker'])
    st.plotly_chart(fig, use_container_width=True) # O gráfico ocupará a largura disponível

    # Seção da Tabela de Trades
    st.subheader("📜 Histórico de Operações")
    trades_df = results['trades']
    if not trades_df.empty:
        # Formatar colunas para melhor visualização
        trades_df_display = trades_df.copy()
        trades_df_display['profit'] = trades_df_display['profit'].map('{:,.2f}'.format).str.replace(",", "X").str.replace(".", ",").str.replace("X", ".")
        trades_df_display['returnPercent'] = trades_df_display['returnPercent'].map('{:.2f}%'.format)
        trades_df_display['shares'] = trades_df_display['shares'].map('{:.4f}'.format) # Mais casas decimais para cripto
        st.dataframe(trades_df_display[[
            'entryDate', 'entryPrice', 'exitDate', 'exitPrice',
            'daysHeld', 'profit', 'returnPercent', 'exitReason'
        ]].style.applymap(lambda v: 'color: green;' if isinstance(v, str) and '-' not in v and ('%' in v or '.' in v) else ('color: red;' if isinstance(v, str) and '-' in v else 'color: black;'), subset=['profit', 'returnPercent'])) # Colore +/-
    else:
        st.info("Nenhuma operação realizada no período.")

# Limpar resultados se o ticker mudar e o botão não for pressionado
elif 'last_ticker' in st.session_state and st.session_state['last_ticker'] != ticker_input:
     st.session_state['backtest_results'] = None

st.sidebar.title("Sobre")
st.sidebar.info(
    """
    Este app simula a estratégia EMA RSI Diário:
    - **Entrada:** Crossover da EMA(2) do RSI(2) de Fechamento sobre a EMA(2) do RSI(2) de Abertura, com EMA(RSI Fechamento) subindo.
    - **Saída:** Crossunder das EMAs OU EMA(RSI Fechamento) caindo OU após 4 dias.
    - **Dados:** Yahoo Finance (Diário).
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido com [Streamlit](https://streamlit.io/)")