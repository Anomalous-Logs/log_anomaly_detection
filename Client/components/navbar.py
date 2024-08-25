import dash
from dash import html,dcc
# import dash_mantine_components as dmc

layout= html.Div([
        # html.H1('Multi-page app with Dash Pages',className="link"),
        # html.Div([
        #     html.Div(
        #         dcc.Link(f"{page['name']} - {page['path']}", href=page["relative_path"])
        #     ) for page in dash.page_registry.values()
        # ]),
        html.H1(["navbar"])
      
        
],className="navbar")