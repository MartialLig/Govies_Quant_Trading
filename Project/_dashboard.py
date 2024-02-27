import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from _data_manager import DataManager
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output


########################################################################################################################################################################################################
################################################### INITIALISE DATASET##################################################################################################################################
########################################################################################################################################################################################################

file_path = "EGB_historical_yield.csv"
data_manager = DataManager(file_path)


########################################################################################################################################################################################################
################################################### CREATION OF GRAPH####################################################################################################################################
########################################################################################################################################################################################################

figures_everything = []
fig = px.line(data_manager.data, x=data_manager.data.index, y=data_manager.data.columns,  # Adjusting column selection
              title=f"Yield Rates ",
              labels={"value": "Yield Rate", "variable": "Country"})
fig.update_layout(title=dict(x=0.5))
figures_everything.append(dcc.Graph(figure=fig))
fig = px.line(data_manager.rank_data, x=data_manager.rank_data.index, y=data_manager.rank_data.columns,  # Adjusting column selection
              title=f"Yield Rates Ranking",
              labels={"value": "Yield Rate", "variable": "Country"},
              markers=True)
fig.update_layout(title=dict(x=0.5))
fig.update_traces(line=dict(width=1))
figures_everything.append(dcc.Graph(figure=fig))
'''fig = px.line(data_manager.spread_yield, x=data_manager.spread_yield.index, y=data_manager.spread_yield.columns,  # Adjusting column selection
              title=f"Yield Spread",
              labels={"value": "Yield Spread Rate", "variable": "Country"})
fig.update_layout(title=dict(x=0.5))
fig.update_traces(line=dict(width=1))
figures_everything.append(dcc.Graph(figure=fig))'''


figures_maturity = []

for maturity, df_maturity in data_manager.data_by_maturity.items():
    fig = px.line(df_maturity, x=df_maturity.index, y=df_maturity.columns,  # Adjusting column selection
                  title=f"Yield Rates Over Time for {maturity} Maturity Bonds",
                  labels={"value": "Yield Rate", "variable": "Country"})
    fig.update_layout(title=dict(x=0.5))
    figures_maturity.append(dcc.Graph(figure=fig))


figures_country = []

for country, df_country in data_manager.data_by_country.items():
    fig = px.line(df_country, x=df_country.index, y=df_country.columns,
                  title=f"Yield Rates Over Time for {country}",
                  labels={"value": "Yield Rate", "variable": "Bond Maturity"})
    fig.update_layout(title=dict(x=0.5))  # Center the title
    figures_country.append(dcc.Graph(figure=fig))

figures_others = []

fig = px.line(data_manager.data_mean_by_country, x=data_manager.data_mean_by_country.index, y=data_manager.data_mean_by_country.columns,  # Adjusting column selection
              title="Average Yield Rates by Country",
              labels={"value": "Average Yield Rate (%)", "variable": "Country"})
fig.update_layout(title=dict(x=0.5))
figures_others.append(dcc.Graph(figure=fig))

fig = px.line(data_manager.data_std_30D_by_country, x=data_manager.data_std_30D_by_country.index, y=data_manager.data_std_30D_by_country.columns,
              title="30-Day Rolling Standard Deviation of Yield Rates by Country",
              labels={"value": "Standard Deviation (%)", "variable": "Country"})
fig.update_layout(title=dict(x=0.5))
figures_others.append(dcc.Graph(figure=fig))

fig = px.line(data_manager.data_mean_by_maturity, x=data_manager.data_mean_by_maturity.index, y=data_manager.data_mean_by_maturity.columns,  # Adjusting column selection
              title="Average Yield Rates by Maturity",
              labels={"value": "Average Yield Rate (%)", "variable": "Maturity"})
fig.update_layout(title=dict(x=0.5))
figures_others.append(dcc.Graph(figure=fig))

fig = px.line(data_manager.data_std_30D_by_maturity, x=data_manager.data_std_30D_by_maturity.index, y=data_manager.data_std_30D_by_maturity.columns,
              title="30-Day Rolling Standard Deviation of Yield Rates by Maturity",
              labels={"value": "Standard Deviation (%)", "variable": "Maturity"})
fig.update_layout(title=dict(x=0.5))
figures_others.append(dcc.Graph(figure=fig))

########################################################################################################################################################################################################
################################################### LAYOUT OF THE DASHBOARD##############################################################################################################################
########################################################################################################################################################################################################

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1(
        'Bond Yield Rates Dashboard',
        style={
            'textAlign': 'center',
            'color': '#007BFF',  # Définit la couleur du texte
            'marginBottom': '20px',  # Ajoute une marge en dessous du titre
            'marginTop': '20px',  # Ajoute une marge au-dessus du titre
            'fontFamily': 'Arial, sans-serif',  # Définit la famille de polices
            'fontWeight': 'bold',  # Rend le texte en gras
        }
    ),

    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Everything', value='tab-0'),
        dcc.Tab(label='Maturity Figures', value='tab-1'),
        dcc.Tab(label='Country Figures', value='tab-2'),
        dcc.Tab(label='Others', value='tab-3'),
    ]),
    html.Div(id='tabs-content')
])


########################################################################################################################################################################################################
################################################### CALLBACK#############################################################################################################################################
########################################################################################################################################################################################################


@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-0':
        return html.Div([
            *figures_everything
        ])
    elif tab == 'tab-1':
        return html.Div([
            *figures_maturity
        ])
    elif tab == 'tab-2':
        return html.Div([
            *figures_country
        ])
    elif tab == 'tab-3':
        return html.Div([
            *figures_others
        ])


########################################################################################################################################################################################################
################################################## LAUNCH DASHBOARD######################################################################################################################################
########################################################################################################################################################################################################
if __name__ == '__main__':
    app.run_server(debug=True)
