from dash import dcc, html

layout = html.Div(
    [
        html.Div(
            [
                dcc.Link("home",href='/'),
                dcc.Link("dashboard",href='/dashboard'),
                dcc.Link("login",href='/login')
            ],className="sidebar_nav"
        ),
    ],className="sidebar"
)


