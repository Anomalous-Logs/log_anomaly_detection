from dash import html, register_page

# Đăng ký trang với Dash
register_page(__name__, path='/login')

layout = html.Div([
    html.H1('login Page'),
    html.P('This is the login page.'),
    html.A('Go to Home Page', href='/'),
    html.Br(),
    html.A('Go to dashboard Page', href='/dashboard')
])
