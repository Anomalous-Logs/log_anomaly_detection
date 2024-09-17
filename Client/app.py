from dash import html,dcc
from components import navbar
from dash import html,dcc, register_page,dash_table
import pandas as pd
import plotly.graph_objs as go
from datetime import date 
from datetime import datetime
from dash.dependencies import Input, Output
from dash import Dash

app = Dash(__name__, suppress_callback_exceptions=True)

# today = date.today()
today = '2023-09-15T12:30:00Z'

df = pd.read_csv('https://raw.githubusercontent.com/logpai/loghub/master/Apache/Apache_2k.log_structured.csv')
df['Time'] = pd.to_datetime(df['Time'])
# dff = df.groupby([df['Level']]).count()
# dff.rename(columns={'LineId':'Level', 'EventId':'Count'}, inplace=True)
df['hours'] = df['Time'].dt.floor('h')
fig=df.groupby(['hours','Level']).size().unstack(fill_value=0)

app.layout = html.Div([
    navbar.layout,
        dcc.Graph(
        id='heatmap',
        figure={
            'data': [
                go.Heatmap(
                    z=fig.T.values,   # Dữ liệu của heatmap (xoay cho đúng chiều)
                    x=fig.index,      # Trục X là Date (thời gian)
                    y=['error','notice'],  # Trục Y là các mức độ log
                    colorscale=[
                         [0, 'lightgreen'],  # Mức thấp (NOTICE)
                         [0.5,'yellow'], 
                        [1, 'red']          # Mức cao (ERROR)
                    ],
                     showscale=True,  # Hiển thị thang màu
                    xgap=2,          # Tạo khoảng cách giữa các ô vuông (chiều ngang)
                    ygap=2,      
                )
            ],
            # 'layout': go.Layout(
            #     title='',
            #     xaxis={'title': 'Time'},
            #     yaxis={'title': 'Level'},
            # )
             'layout': go.Layout(
                title='',
                xaxis={
                    'title': 'Time (Grouped by Hour)',
                    'tickvals': fig.index,  # Hiển thị nhãn cho từng giờ
                    'ticktext': fig.index.strftime('%Y-%m-%d %H:%M'),  # Định dạng hiển thị của nhãn
                    # 'tickangle': -45  # Xoay nhãn thời gian để dễ đọc hơn
                },
                yaxis={'title': 'Level'},
               
                
    
            )
        }),
    html.Div([
        dcc.Input(
        id='input-search',
        type='search',
        value='',
        placeholder='Search by name',
        maxLength=50,
        debounce=True,
    ),
    dcc.DatePickerSingle(
        id='date-picker',
        date=today
    ),
        dcc.Input(
        id='input-Severity',
        type='text',
        value='',
        placeholder='Severity',
        maxLength=50,
        debounce=True,
    ),
        dcc.Input(
        id='input-Interval',
        type='text',
        value='',
        placeholder='Interval',
        maxLength=50,
        debounce=True,
    ),

    ],className="home__inp"),
    html.Div([
        dash_table.DataTable(
        df.to_dict('records'),
        [{"name": i, "id": i} for i in df.columns]
    )
    ],className="home__table")
],className='home_container'),

# Callback để in giá trị ngày được chọn
@app.callback(
    Input('date-picker', 'date')
)
def update_output(date):
    dt = datetime.fromisoformat(date)
    rfc3339_time = dt.isoformat() + "Z"
    print(f'Ngày được chọn: {rfc3339_time}')


if __name__ == '__main__':
    app.run_server(debug=True)