from dash import html,dcc, register_page,dash_table
import plotly.express as px
import pandas as pd

# Đăng ký trang với Dash
register_page(__name__, path='/')

df = pd.read_csv('https://raw.githubusercontent.com/logpai/loghub/master/Apache/Apache_2k.log_structured.csv')

# Tạo một biểu đồ
fig = px.line(df, x="Time", y="Level", title="Biểu đồ ví dụ")
# Tạo bảng dữ liệu
table = dash_table.DataTable(
    columns=[{"name": col, "id": col} for col in df.columns],
    data=df.to_dict('records'),
    style_table={'overflowX': 'auto'},
    style_cell={'textAlign': 'left'}
)

layout = html.Div([
    dcc.Graph(figure=fig),
    html.Div([
        dcc.Input(
        id='input-box',
        type='search',
        value='',
        placeholder='Search by name',
        maxLength=50,
        debounce=True,
    ),
        dcc.Input(
        id='input-box',
        type='text',
        value='',
        placeholder='Severity',
        maxLength=50,
        debounce=True,
    ),
        dcc.Input(
        id='input-box',
        type='text',
        value='',
        placeholder='Interval',
        maxLength=50,
        debounce=True,
    ),
        dcc.DatePickerSingle(
        id='date-picker',
        date='2024-08-18'
    ),
    ],className="home__inp"),
    html.Div([
        table
    ],className="home__table")
])
