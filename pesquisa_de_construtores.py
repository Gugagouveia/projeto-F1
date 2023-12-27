# -*- coding: utf-8 -*-
"""Importando as Bibliotecas"""

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import time
import datetime
import scipy.stats as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

"""Importando os CSVs"""

circuits = pd.read_csv('/content/circuits.csv')

constructor_results = pd.read_csv('/content/constructor_results.csv')

constructor_standings = pd.read_csv('/content/constructor_standings.csv')

constructors = pd.read_csv('/content/constructors.csv')

driver_standings = pd.read_csv('/content/driver_standings.csv')

drivers = pd.read_csv('/content/drivers.csv')

lap_times = pd.read_csv('/content/lap_times.csv')

pit_stops = pd.read_csv('/content/pit_stops.csv')

qualifying = pd.read_csv('/content/qualifying.csv')

races = pd.read_csv('/content/races.csv')

results = pd.read_csv('/content/results.csv')

seasons = pd.read_csv('/content/seasons.csv')

sprint_results = pd.read_csv('/content/sprint_results.csv')

status = pd.read_csv('/content/status.csv')

"""Pilotos"""

print(driver_standings)

"""Mergeando dataframes"""

team = constructors.merge(results,on='constructorId',how = 'left')

print(team.columns)

"""Agrupando a coluna por nome do Construtor"""

best = team[['name','points','raceId']]
best = best.groupby('name')['raceId'].nunique().sort_values(ascending=False).reset_index(name = 'races')
best = best[best['races'] >= 100]
best.head()

""" Calculando pontos por corrida"""

func = lambda x: x.points.sum()/x.raceId.nunique()
data = team[team['name'].isin(best.name)].groupby('name').apply(func).sort_values(ascending=False).reset_index(name = 'pontos por corrida')
data.head(10)

"""Exbindo os Resultados"""

fig = go.Figure(
    data=[go.Bar(x = data.name, y=data['pontos por corrida'])],
    layout_title_text="Construtores com mais Pontos por corrida"

)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.update_traces(textfont_size=20,
                  marker=dict(line=dict(color='#000000', width=2)))
fig.show()

"""O Estudo a cima foi feito em cima das contrutoras, em seguida faremos o estudo em cima dos pilotos."""

concat_driver_name = lambda x: f"{x.forename} {x.surname}"

drivers['driver'] = drivers.apply(concat_driver_name, axis=1)

"""Montando Histórico de Vitórias dos 5 maiores Campeões."""

results_copy = results.set_index('raceId').copy()
races_copy = races.set_index('raceId').copy()

results_copy = results_copy.query("position == '1'")
results_copy['position'] = 1

results_cols = ['driverId', 'position']
races_cols = ['date']
drivers_cols = ['driver', 'driverId']

results_copy = results_copy[results_cols]
races_copy = races_copy[races_cols]
drivers_copy = drivers[drivers_cols]

f1_victories = results_copy.join(races_copy)
f1_victories = f1_victories.merge(drivers_copy, on='driverId', how='left')


f1_victories = f1_victories.sort_values(by='date')

f1_victories['victories'] = f1_victories.groupby(['driverId']).cumsum()


f1_biggest_winners = f1_victories.groupby('driverId').victories.nlargest(1).sort_values(ascending=False).head(5)
f1_biggest_winners_ids = [driver for driver, race in f1_biggest_winners.index]

f1_victories_biggest_winners = f1_victories.query(f"driverId == {f1_biggest_winners_ids}")

results_copy = results.set_index('raceId').copy()
races_copy = races.set_index('raceId').copy()

results_copy = results_copy.query("position == '1'")
results_copy['position'] = 1

results_cols = ['driverId', 'position']
races_cols = ['date']
drivers_cols = ['driver', 'driverId']

results_copy = results_copy[results_cols]
races_copy = races_copy[races_cols]
drivers_copy = drivers[drivers_cols]

f1_victories = results_copy.join(races_copy)
f1_victories = f1_victories.merge(drivers_copy, on='driverId', how='left')

f1_victories = f1_victories.sort_values(by='date')

f1_victories['victories'] = f1_victories.groupby(['driverId']).cumsum()

f1_biggest_winners = f1_victories.groupby('driverId').victories.nlargest(1).sort_values(ascending=False).head(5)
f1_biggest_winners_ids = [driver for driver, race in f1_biggest_winners.index]

f1_victories_biggest_winners = f1_victories.query(f"driverId == {f1_biggest_winners_ids}")

cols = ['date', 'driver', 'victories']
winner_drivers = f1_victories_biggest_winners.driver.unique()

colors = {
    'Alain Prost': '#d80005',
    'Ayrton Senna': '#ffffff',
    'Michael Schumacher': '#f71120',
    'Sebastian Vettel': '#10428e',
    'Lewis Hamilton': '#e6e6e6'
}

winners_history = pd.DataFrame()

for driver in winner_drivers:

    driver_history = f1_victories_biggest_winners.query(f"driver == '{driver}'")[cols]

    other_drivers = winner_drivers[winner_drivers != driver]
    other_drivers = list(other_drivers)

    other_driver_history = f1_victories_biggest_winners.query(f"driver == {other_drivers}")[cols]

    other_driver_history['driver'] = driver

    other_driver_history['victories'] = 0

    driver_history = pd.concat([driver_history, other_driver_history])

    driver_history.sort_values(by='date', inplace=True)


    driver_history.reset_index(inplace=True)


    for index, row in driver_history.iterrows():
        if not row['victories'] and index-1 > 0:
            driver_history.loc[index, 'victories'] = driver_history.loc[index-1, 'victories']

    winners_history = pd.concat([winners_history, driver_history])

"""Exibindo resultados"""

fig = go.Figure()

fig = px.bar(
    winners_history,
    x='victories',
    y='driver',
    color='driver',
    orientation='h',
    animation_frame="date",
    animation_group="driver",
)

fig.update_traces(dict(marker_line_width=1, marker_line_color="black"))

fig.update_layout(xaxis=dict(range=[0, 100]))

fig.update_layout(title_text="Vitórias em corridas na história da F1 entre os 5 melhores pilotos.")

fig.update_layout(
    updatemenus = [
        {
            "buttons": [
                # Play
                {
                    "args": [
                        None,
                        {
                            "frame": {
                                "duration": 100,
                                 "redraw": False
                            },
                            "fromcurrent": True,
                            "transition": {
                                "duration": 100,
                                "easing": "linear"
                            }
                        }
                    ],
                    "label": "Play",
                    "method": "animate"
                },
                # Pause
                {
                    "args": [
                        [None],
                        {
                            "frame": {
                                "duration": 0,
                                "redraw": False
                            },
                            "mode": "immediate",
                            "transition": {
                                "duration": 0
                            }
                        }
                    ],
                    "label": "Pausa",
                    "method": "animate"
                }
            ]
        }
    ]
)

fig.show()

"""Montando Gráfico de Pole Positions."""

winner_drivers_ids = f1_victories_biggest_winners[['driverId', 'driver']].drop_duplicates()
winner_drivers_map = {}

for _, row in winner_drivers_ids.iterrows():
    winner_drivers_map[row['driverId']] = row['driver']

f1_biggest_winners_poles = results.query(f"driverId == {f1_biggest_winners_ids} & grid == 1")[['driverId', 'grid']]

f1_biggest_winners_poles['driver'] = f1_biggest_winners_poles.driverId.map(winner_drivers_map)
f1_biggest_winners_poles['color'] = f1_biggest_winners_poles.driver.map(colors)

f1_biggest_winners_poles['total_poles'] = f1_biggest_winners_poles.groupby(['driverId']).cumsum()

f1_biggest_winners_total_poles = f1_biggest_winners_poles.groupby('driver').total_poles.nlargest(1).sort_values(ascending=False).head(5)
f1_biggest_winners_total_poles = pd.DataFrame(f1_biggest_winners_total_poles).reset_index()

f1_biggest_winners_total_poles['color'] = f1_biggest_winners_total_poles.driver.map(colors)

fig = px.bar(
    f1_biggest_winners_total_poles,
    x='driver',
    y='total_poles',
    color='driver',
    color_discrete_sequence=f1_biggest_winners_total_poles.color
)

fig.update_traces(dict(marker_line_width=1, marker_line_color="black"))

fig.update_layout(title_text="Pole positions entre os 5 melhores pilotos da história")

fig.show()

"""Montando datadset do Hamilton"""

hamilton = drivers.query("driverRef == 'hamilton'")

"""Dataframe das corridas"""

def get_races_by_driver_id(driver_id):
    columns = ['grid', 'position', 'raceId', 'constructorId', 'statusId']

    driver_races = results.query(f'driverId == {driver_id}')
    driver_races = driver_races[columns]

    driver_races.set_index('raceId', inplace=True)

    driver_races = driver_races.join(races.set_index('raceId')['date'])

    driver_races['is_pole'] = driver_races.grid == 1
    driver_races['is_first_place'] = driver_races.position == '1'

    driver_races.sort_values(by='date', inplace=True)

    driver_races['poles'] = driver_races.is_pole.cumsum()
    driver_races['races_won'] = driver_races.is_first_place.cumsum()

    driver_races = driver_races.set_index('constructorId').join(constructors.set_index('constructorId')['name'])
    driver_races = driver_races.rename(columns={'name': 'constructor'})

    driver_races = pd.merge(status, driver_races, on=['statusId', 'statusId']).sort_values(by='date')
    driver_races = driver_races.rename(columns={'status': 'race_status'})

    return driver_races

hamilton_races = get_races_by_driver_id(hamilton.driverId[0])

"""Separando as Construtoras"""

mc_laren = hamilton_races.query('constructor == "McLaren"')
mercedes = hamilton_races.query('constructor == "Mercedes"')

mc_laren = pd.concat([mc_laren, mercedes.head(1)])

"""# Pole positions x Corridas"""

mclaren_poles  = go.Scatter(x=mc_laren.date, y=mc_laren.poles, fill='tozeroy', name="Mc Laren", marker=dict(color="#D89A8C"))
mercedes_poles = go.Scatter(x=mercedes.date, y=mercedes.poles, fill='tozeroy', name="Mercedes", marker=dict(color="#C2C2C2"))

mclaren_wons  = go.Scatter(x=mc_laren.date, y=mc_laren.races_won, fill='tozeroy', name="Mc Laren", marker=dict(color="#cb7967"), showlegend=False)
mercedes_wons = go.Scatter(x=mercedes.date, y=mercedes.races_won, fill='tozeroy', name="Mercedes", marker=dict(color="#b3b3b3"), showlegend=False)

fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=("Pole positions","Corridas Ganhas")
)

fig.add_trace(mclaren_poles, row=1, col=1)
fig.add_trace(mercedes_poles, row=1, col=1)

fig.add_trace(mclaren_wons, row=2, col=1)
fig.add_trace(mercedes_wons, row=2, col=1)

fig.update_layout(
    height=600,
    title_text="Vitórias ao longo da carreira",
    title_font_size=20,
    hovermode='x',
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.95,
        xanchor="left",
        x=0.01
    ),
)

fig.update_yaxes(range=[0, 100])

"""###Conclusões
Hamilton tem um número impressionante de vitórias em corridas e pole positions em comparação com os melhores pilotos de F1 (vencedores de corridas e campeonatos)
A sua carreira na Mercedes é determinante para os seus números até ao momento: foi nesse construtor que ele teve mais crescimento de vitórias em corridas.
No período de Nico Rosberg, Hamilton tinha um companheiro de equipe mais competitivo, com números mais próximos entre eles.
No período Valtteri Bottas vemos um trabalho de equipe muito bom para os resultados de Hamilton. Melhoria na classificação e quase todas as corridas concluídas.

"""

# Unificar as tabelas com base em uma coluna comum
tabela_races_circuits = pd.merge(races, circuits, on='circuitId')
# Remover colunas indesejadas
colunas_indesejadas = ['url_x','fp1_date','fp1_time','fp2_date','fp2_time','fp3_date','fp3_time','quali_date','quali_time',
                       'sprint_date','sprint_time','lat','lng','alt','url_y']
tabela_races_circuits = tabela_races_circuits.drop(columns=colunas_indesejadas)
# Ajustar o nome das colunas
tabela_races_circuits.rename(columns={'name_x': 'name_races', 'name_y': 'name_circuits'}, inplace=True)

# Unificar as tabelas com base em uma coluna comum
tabela_pitstops_laptimes = pd.merge(pit_stops, lap_times, on=['raceId','driverId','lap'])
# Ajustar o nome das colunas
tabela_pitstops_laptimes.rename(columns={'time_x':'time_pitstop','milliseconds_x':'milliseconds_pitstop',
                                      'time_y':'time_laptime','milliseconds_y':'milliseconds_laptime','duration':'duration_pitstop'},
                             inplace=True)

# Unificar as tabelas com base em uma coluna comum
tabela_rc_pl = pd.merge(tabela_races_circuits, tabela_pitstops_laptimes, on='raceId')

# Remover colunas indesejadas
colunas_indesejadas = ['number','code','dob','nationality','url']
tabela_drivers = drivers.drop(columns=colunas_indesejadas)

tabela_rc_pl_d = pd.merge(tabela_rc_pl, tabela_drivers, on='driverId')
tabela_rc_pl_d_cs = pd.merge(tabela_rc_pl_d, constructor_standings, on='raceId')

# Unificar as tabelas com base em uma coluna comum
tabela_final = pd.merge(tabela_rc_pl_d_cs, constructors, on='constructorId')
###
# Tabela tabela_final (Geral - Year)
# Aplicar um filtro (year)
filtro = tabela_final['year'] > 2010
tabela_final_year = tabela_final[filtro]

# Aplicar um filtro (circuitId)
filtro = tabela_final['circuitId'] == 6
tabela_monaco = tabela_final[filtro]

# Aplicar um filtro (year)
filtro = tabela_monaco['year'] > 2020
tabela_monaco_year = tabela_monaco[filtro]

print(tabela_final.head())

# Durações de pit stop (em segundos)
pitstop_duration_geral = np.array(tabela_final['duration_pitstop'])

total_seconds_geral = []

for value in pitstop_duration_geral:
    if len(value.split(':')) > 1:
        minutes,seconds = value.split(':')
        minutes = int(minutes)
        seconds = int(seconds.split('.')[0])
        total_seconds_geral.append((minutes * 60) + seconds)
    else:
        seconds = int(value.split('.')[0])
        total_seconds_geral.append(seconds)

# Exibindo os melhores e piores resultados
print("Melhor tempo: ", int(min(total_seconds_geral)), "segundos")
print("Pior tempo: ", int(max(total_seconds_geral)), "segundos")

# Cálculo da média
mean_duration_geral = np.mean(total_seconds_geral)
print("Duração média do pit stop:", int(mean_duration_geral), "segundos")

# Cálculo da variação (desvio padrão)
std_dev_geral = np.std(total_seconds_geral)
print(f"\nVariação da duração do pit stop: {int(std_dev_geral)} segundos")

# Cria o histograma
plt.figure(figsize=(10,6)) # Tamanho do gráfico
plt.hist(total_seconds_geral, bins=50, density=True, alpha=0.75, color='b')

# Adiciona título e rótulos
plt.title('Distribuição das Durações dos Pit Stops(Mercedes)')
plt.xlabel('Duração de Pit Stop (segundos)')
plt.ylabel('Frequência')

# Adiciona uma grade
plt.grid(True)

# Adiciona uma legenda
plt.legend(['Duração de pit-stops em segundos'])

# Mostra o gráfico
plt.show()

def getTotalSeconds(param_time_monaco):
    return_total = []

    for value in param_time_monaco:
        if len(value.split(':')) > 1:
            minutes,seconds = value.split(':')
            minutes = int(minutes)
            seconds = int(seconds.split('.')[0])
            return_total.append((minutes * 60) + seconds)
        else:
            seconds = int(value.split('.')[0])
            return_total.append(seconds)

    return return_total

# Durações de pit stop (em segundos)
pitstop_duration_monaco = np.array(tabela_monaco['duration_pitstop'])

total_seconds_monaco = getTotalSeconds(pitstop_duration_monaco)

# Tempo da Última Volta (em segundos)
time_laptime_monaco = np.array(tabela_monaco['time_laptime'])

total_time_laptime_monaco = getTotalSeconds(time_laptime_monaco)

# Exibindo os melhores e piores resultados
print("Melhor tempo: ", int(min(total_seconds_monaco)), "segundos")
print("Pior tempo: ", int(max(total_seconds_monaco)), "segundos")

# Cálculo da média
mean_duration_monaco = np.mean(total_seconds_monaco)
print("Duração média do pit stop:", int(mean_duration_monaco), "segundos")

# Cálculo da variação (desvio padrão)
std_dev_monaco = np.std(total_seconds_monaco)
print(f"\nVariação da duração do pit stop: {int(std_dev_monaco)} segundos")

# Cria o histograma
plt.figure(figsize=(10,6)) # Tamanho do gráfico
plt.hist(total_seconds_monaco, bins=50, density=True, alpha=0.75, color='b')

# Adiciona título e rótulos
plt.title('Distribuição das Durações dos Pit Stops(Construtora Rival )')
plt.xlabel('Duração de Pit Stop (segundos)')
plt.ylabel('Frequência')

# Adiciona uma grade
plt.grid(True)

# Adiciona uma legenda
plt.legend(['Duração de pit-stops em segundos'])

# Mostra o gráfico
plt.show()