import pandas as pd
import plotly.graph_objs as go 
import dash_html_components as html 
from dash.dependencies import Input, Output 
import dash_core_components as dcc 
from dash import Dash, html 
import dash_bootstrap_components as dbc
import dash 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Import File
df = pd.read_csv('C:/DSS/venv/DSSFINALPROJECT.csv',sep=',')
df2 = pd.read_csv('C:/DSS/venv/Cleaned-job.csv',sep=',')
df3 = pd.read_csv('C:/DSS/venv/Cleaned-poutcome.csv',sep=',')

# ============================ df =============================== #

# Create a cross-tabulation table for education and job
job_y_ct = pd.crosstab(df['job'], df['y'])
edu_y_ct = pd.crosstab(df['education'], df['y'])

# Create the layout for the first plot
layout1 = go.Figure(data=[go.Pie(labels=job_y_ct.index)])
# Create the layout for the second plot
layout2 = go.Figure(data=[go.Pie(labels=edu_y_ct.index)])


# =============================================================== #

# =========================== df2 =============================== #
# Define the job types and load the job data

job_types = df2['job'].unique()
job_data = {job: df2[df2['job'] == job] for job in job_types}

# ================================================================ #

# Create a dropdown menu of unique job values
job_options = [{'label': job, 'value': job} for job in df['job'].unique()]

# Create a dropdown menu of unique edu values
edu_options = [{'label': edu, 'value': edu} for edu in df['education'].unique()]

# ================================================================ #

# All Graph
job_pie_graph = dcc.Graph(
                    id='job-pie-graph',
                    figure=layout1,
                    className='chart',)

edu_pie_graph = dcc.Graph(
                    id='edu-pie-graph',
                    figure=layout2,
                    className='chart',)

accuracy_graph = dcc.Graph(
                    id='accuracy-graph',
                    className='chart',)

prescriptive_graph = dcc.Graph(
                    id='mean-propertion-graph',
                    className='chart',)

diagnostic_graph = dcc.Graph(
                    id='diagnostic-graph',
                    className='chart')

diagnostic_graph2 = dcc.Graph(
                    id='diagnostic-graph2',
                    className='chart')

# ================================================================ #
# Dropdown Menu for Job and Education
job_drop_menu = dcc.Dropdown(
                    id='job-dropdown',
                    options=job_options,
                    value=df['job'].unique()[0],
                    className='dropdown',
                    searchable=False,
                    clearable=False
                )
edu_drop_menu = dcc.Dropdown(
                    id='edu-dropdown',
                    options=edu_options,
                    value=df['education'].unique()[0],
                    className='dropdown',
                    searchable=False,
                    clearable=False
)
# ================================================================ #

# All Tittle
title1 = "Descriptive Analytics by Job Status"
JobTitle = html.H3(children=title1, className='title')

title2 = "Descriptive Analytics by Education Level"
EduTitle = html.H3(children=title2, className='title')

title3 = "Predictive Analytics by Job Status"
PredAnTitle = html.H3(children=title3, className='title')

title4 = "Prescriptive Analytics by Job Status"
PresAnTitle = html.H3(children=title4, className='title')

title5 = "Diagnostic Analytics by Previous Outcome (y==1)"
DiagAnTitle1 = html.H3(children=title5, className='title')

title6 = "Diagnostic Analytics by Previous Outcome (y==0)"
DiagAnTitle2 = html.H3(children=title6, className='title')

# ================================================================ #

# navbar for ("Web-Based Dashboard")
navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("Web-Based Dashboard by Group 2", className="ml-2"),
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,   
)
# ================================================================ #

# Create a new container with the group name text
group_container = dbc.Container([
    dbc.Row([
        dbc.Col(html.H5(f"Group Members:")),
    ]),
    dbc.Row([
        dbc.Col(html.P(f"Ayorbaba, Ferlien")),
    ]),
    dbc.Row([
        dbc.Col(html.P(f"Harbas, Ribka")),
    ]),
    dbc.Row([
        dbc.Col(html.P(f"Mangi, Joanne")),
    ]),
    dbc.Row([
        dbc.Col(html.P(f"Sarfunin, Maria")),
    ]),
    dbc.Row([
        dbc.Col(html.P(f"Tuuk, Jennyfer")),
    ]),
])

# Layout Dashboard
app.layout = dbc.Container([
    navbar,
    group_container,
    dbc.Row([
        dbc.Col(html.H3(children=[JobTitle, job_drop_menu, job_pie_graph]), width=6),
        dbc.Col(html.H3(children=[EduTitle, edu_drop_menu, edu_pie_graph]), width=6),
        dbc.Col(html.H3(children=[PredAnTitle, accuracy_graph]), width=6),
        dbc.Col(html.H3(children=[PresAnTitle, prescriptive_graph]), width=6),
        dbc.Col(html.H3(children=[DiagAnTitle1, diagnostic_graph]), width=6),
        dbc.Col(html.H3(children=[DiagAnTitle2, diagnostic_graph2]), width=6)
    ])
],fluid=True)

# ================================================================ #
 
# def Descriptive Analytics Graph by Job Status 
def update_chart_job(selected_job):
    # Filter the dataframe to only include rows with the selected job value
    filtered_df_job = df[df['job'] == selected_job]
    # Create a new cross-tabulation with only the 'subscribed' and 'not subscribed' columns
    job_y_ct = pd.crosstab(filtered_df_job['y'], columns='count')
    job_y_ct.index = ['no', 'yes']
    # Define the colors for each slice of the pie chart
    colors = ['#FDE2F3', '#9F67FF']
    # Create a new pie chart with the specified colors and depth
    layout1 = go.Figure(data=[go.Pie(labels=job_y_ct.index, values=job_y_ct['count'], 
                                 marker={'colors': colors}, hole=0.4)])
    layout1.update_layout(
                    showlegend=True,
                    legend={'x': 1, 'y': 0.5,
                            'title': {'text': 'Deposit', 'font': {'family': 'sans-serif', 'size': 30, 'color': '#2A2F4F'}},
                            'font': {'family': 'sans-serif', 'size': 25, 'color': '#2A2F4F'}})

    layout1.update_traces(hoverinfo='label + percent', textinfo='value', textfont_size=20,
                      marker=dict(colors=colors, line=dict(color='#000000', width=1)))
    return layout1
# ==================================================================================================== #

# def Descriptive Analytics Graph by Education Status
def update_chart_edu(selected_edu):
    # Filter the dataframe to only include rows with the selected education value
    filtered_df_edu = df[df['education'] == selected_edu]
    # Create a new cross-tabulation with only the 'subscribed' and 'not subscribed' columns
    edu_y_ct = pd.crosstab(filtered_df_edu['y'], columns='count')
    edu_y_ct.index = ['no', 'yes']
    # Define the colors for each slice of the pie chart
    colors = ['#9F67FF', '#FDE2F3']
    # Create a new pie chart with the specified colors and depth
    layout2 = go.Figure(data=[go.Pie(labels=edu_y_ct.index, values=edu_y_ct['count'], 
                                 marker={'colors': colors}, hole=0.4)])
    layout2.update_layout(
        showlegend=True,
        legend={
            'x': 1,
            'y': 0.5,
            'title': {'text': 'Deposit', 'font': {'family': 'sans-serif', 'size': 30, 'color': '#2A2F4F'}},
            'font': {'family': 'sans-serif', 'size': 25, 'color': '#2A2F4F'}
        })
    layout2.update_traces(hoverinfo='label + percent', textinfo='value', textfont_size=20,
                      marker=dict(colors=colors, line=dict(color='#000000', width=1)))
    return layout2
# ==================================================================================================== #

# def Predictive Analytics Graph by Job Status 
def update_graph(_):
    # Create empty lists for x and y values
    x_vals = []
    y_vals = []

    for job in job_types:
        # Get the subset of data for the job type
        subset = job_data[job]

        # Split the subset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(subset[['job_status', 'unemployed', 'services', 'management',
                                                                    'blue-collar', 'self-employed', 'technician', 'entrepreneur',
                                                                    'admin.', 'student', 'housemaid', 'retired', 'unknown']], 
                                                            subset['y'], test_size=0.3, random_state=42)

        # Create and fit a decision tree classifier model
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        # Predict output variable 'y' for the testing set
        y_pred = model.predict(X_test)

        # Evaluate model accuracy
        accuracy = model.score(X_test, y_test)

        # Append x and y values to lists
        x_vals.append(job)
        y_vals.append(accuracy)


     # Colors
    color = ['#6795FF','#6780FF','#676CFF','#7667FF','#8A67FF','#9F67FF',
             '#B367FF','#C767FF','#DC67FF','#F067FF','#FF67FA','#FF67E6']   
    # Create a bar plot of model accuracies
    data = [go.Bar(x=x_vals, 
                   y=y_vals,
                   marker={'color': color})]

    layout3 = go.Layout(
                       xaxis=dict(title='Job Type'),
                       yaxis=dict(title='Model Accuracy'),
                       barmode='stack',
                       )
    layout3 = go.Figure(data=data, layout=layout3)

    return layout3
# ==================================================================================================== #

# def Prescriptive Analytics Graph by Mean from Job_status
def update_graph2(_):
    # Group data by job type and calculate mean of output variable 'y'
    job_stats = df2.groupby('job_status')['y'].agg(['count','mean','std'])

    color = ['#6795FA','#6780FA','#676CFA','#7667FA','#8A67FA','#9F67FA',
             '#B367FA','#C767FA','#DC67FA','#F067FA','#FA67FA','#FA67E6']
    
    # Create bar chart
    data = [go.Bar(x=job_stats.index, y=job_stats['mean'],
                   marker={'color': color})]
    layout4 = go.Layout(
                       xaxis=dict(title='Job Type'),
                       yaxis=dict(title='Proportion of Yes Responses'))
    layout4 = go.Figure(data=data, layout=layout4)

    return layout4
# ==================================================================================================== #

# def Diagnostic Analytics Graph by Previous Outcome (y==1)
def update_graph3(_):
    poutcome_success = df3[df3['y'] == 1].groupby('poutcome_status').size() / df3.groupby('poutcome_status').size()
    color = ['#6795FF','#8A67FF','#DC67FF','#FF67E6']
    data = [go.Bar(x=poutcome_success.index,
                   y=poutcome_success.values,
                   marker={'color': color})]
    layout5 = go.Layout(
                        xaxis=dict(title='Previous Campaign Outcome'),
                        yaxis=dict(title='Success Rate'))
    layout5 = go.Figure(data=data, layout=layout5)
    
    return layout5
# ==================================================================================================== #

# def Diagnostic Analytics Graph by Previous Outcome (y==0) 
def update_graph4(_):
    poutcome_success = df3[df3['y'] == 0].groupby('poutcome_status').size() / df3.groupby('poutcome_status').size()
    
    color = ['#FF67FA','#B367FF','#7667FF','#6780FF']
    data = [go.Bar(x=poutcome_success.index,
                   y=poutcome_success.values,
                   marker={'color': color})]
    layout6 = go.Layout(
                        xaxis=dict(title='Previous Campaign Outcome'),
                        yaxis=dict(title='Failure Rate'))
    layout6 = go.Figure(data=data, layout=layout6)
    
    return layout6
# ==================================================================================================== #
   
# Callback Graph 1 - 6
@app.callback(Output('accuracy-graph', 'figure'), 
              [Input('accuracy-graph', 'id')])
def update_graph_acc(_):
    layout3 = update_graph(_)
    return layout3

@app.callback(Output('mean-propertion-graph','figure'),
              [Input('mean-propertion-graph','id')])
def update_graph_mean(_):
    layout4 = update_graph2(_)
    return layout4

@app.callback(Output('diagnostic-graph','figure'),
              [Input('diagnostic-graph','id')])
def update_graph_diagnostic(_):
    layout5 = update_graph3(_)
    return layout5

@app.callback(Output('diagnostic-graph2','figure'),
              [Input('diagnostic-graph2','id')])
def update_graph_diagnostic2(_):
    layout6 = update_graph4(_)
    return layout6

@app.callback(Output('job-pie-graph', 'figure'),
              [Input('job-dropdown', 'value')])
def update_pie_chart(selected_job):
    layout1 = update_chart_job(selected_job)
    return layout1

@app.callback(Output('edu-pie-graph', 'figure'),
              [Input('edu-dropdown', 'value')])
def update_pie_chart(selected_edu):
    layout2 = update_chart_edu(selected_edu)
    return layout2
# ==================================================================================================== #

if __name__ == '__main__':
    app.run_server(debug=True)