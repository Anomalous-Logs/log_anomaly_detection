import dash_html_components as html

def breadcrumb(pathname):
    path_parts = pathname.strip('/').split('/')
    breadcrumb_links = ['Home'] + path_parts
    return html.Div([' > '.join(breadcrumb_links)])