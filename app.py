
from dash import Dash, html,dash_table,dcc,Input,Output,callback
import pandas as pd
import sqlite3
import plotly.express as px

#tailwind config
external_script = ["https://tailwindcss.com/", {"src": "https://cdn.tailwindcss.com"}]

# Connect to the SQLite database
conn = sqlite3.connect("project.db")

# Read the entire 'projects' table into a DataFrame
df = pd.read_sql("SELECT * FROM projects", conn)


#first three boxes
total_project_count = df['Project Title'].nunique()
total_domain_count = df['Domain'].str.split(',').explode().nunique()


#domain wise analytics
#overall analytics
domains = df['Domain'].str.split(',').explode().str.strip()
domains_counts = domains.value_counts()
pie = px.pie( values=domains_counts.values,names=domains_counts.index)

#branch wise analytics




# Create a Dash app
app = Dash(__name__,external_scripts=external_script)

app.layout = html.Div(
    className='bg-[#f1f1fb] min-h-screen flex flex-col',
    children=[
        html.Div(
            className='text-3xl p-4 font-semibold ',
            children='Dashboard for Mini Project Data Analysis'
        ),
        html.Div(
            className='flex justify-around ',
            children=[
                html.Div(
                    className='w-[30%] h-[25vh]  rounded-lg flex flex-col items-center p-4 bg-white',
                    children=[
                        html.Div( className='text-xl h-[30%] flex items-center  font-semibold',children='Total number of projects created: '),
                        html.Div(className='text-6xl h-[70%]  flex items-center',children=total_project_count)
                    ]
                ),
                html.Div(
                    className='w-[30%] h-[25vh]  rounded-lg flex flex-col items-center p-4 bg-white',
                    children=[
                        html.Div( className='text-xl h-[30%] flex items-center  font-semibold',children='Number of Unique Domains Explored: '),
                        html.Div(className='text-6xl h-[70%]  flex items-center',children=total_domain_count)
                    ]
                ),
                html.Div(
                    className='w-[30%] h-[25vh]  rounded-lg flex flex-col items-center p-4 bg-white',
                    children=[
                        html.Div( className='text-xl h-[30%] flex items-center  font-semibold',children='Number of Unique Domains Explored: '),
                        html.Div(className='text-6xl h-[70%]  flex items-center',children=total_domain_count)
                    ]
                ),
            ]
        ),
        html.Div(
            className='p-6 flex flex-col gap-6',
            children=[
                html.Div(className='text-2xl font-bold',children='Domain Wise analysis of Projects:'),
                html.Div(
                    className='flex flex-1 justify-around min-h-[50vh] ',
                    children=[
                        dcc.Graph(className='w-[50%] rounded-lg border-5 border-black ',figure=pie),
                        html.Div(
                            children=[dcc.RadioItems(
                                ['Comps','CSE-AIML','CSE-DS'],
                                'Linear',
                                id='crossfilter-xaxis-type',
                                labelStyle={'display': 'inline-block', 'marginTop': '5px','paddingLeft':'2rem'}
                            )]
                        )
                    ]
                ),
            ]   
        )

    ]
)
# Run the app
app.css.config.serve_locally = True

if __name__ == '__main__':
    app.run_server(debug=True)