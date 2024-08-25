from dash import html, register_page

# Đăng ký trang với Dash
register_page(__name__,)

layout = html.Div([
    html.H1('dashboard Page'),
    html.P('This is the dashboard page.'),
    html.A('Go to Home Page', href='/'),
    html.Br(),
    html.A('Go to login Page', href='/login')
])
