import dash
from dash import Dash, html,dcc
from components import navbar
from components import sidebar 

app = Dash(__name__, use_pages=True)


app.layout = html.Div([
    sidebar.layout,
    html.Div([
        navbar.layout,
        dash.page_container
    ],className="app_container"),
],className="app")



if __name__ == '__main__':
    app.run(debug=True)